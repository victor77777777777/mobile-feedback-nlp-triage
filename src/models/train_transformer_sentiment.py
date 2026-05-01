import os
import json
import argparse
import inspect
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# =========================
# 1. 固定随机种子，保证结果尽量可复现
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# 2. 带类别权重的 Trainer
#    目的：缓解 neutral 类样本较少的问题
# =========================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights,
                dtype=torch.float
            ).to(self.model.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


# =========================
# 3. 读取和抽样数据
# =========================
def load_and_prepare_data(
    data_path: str,
    text_col: str = "review_text",
    label_col: str = "sentiment_label",
    max_per_class: int = 10000,
    seed: int = 42,
):
    print("\n========== Loading data ==========")
    df = pd.read_csv(data_path)

    required_cols = [text_col, label_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    df = df[[text_col, label_col]].copy()
    df = df.dropna(subset=[text_col, label_col])

    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]

    # 只保留三类情感标签
    valid_labels = ["negative", "neutral", "positive"]
    df = df[df[label_col].isin(valid_labels)]

    print(f"Original usable data size: {len(df)}")
    print("\nOriginal label distribution:")
    print(df[label_col].value_counts())

    # 每类最多抽样 max_per_class 条
    # 这样第一次训练不会太慢，也不会让 positive 类过度主导
    sampled_df = (
        df.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=seed))
        .reset_index(drop=True)
    )

    print(f"\nSampled data size: {len(sampled_df)}")
    print("\nSampled label distribution:")
    print(sampled_df[label_col].value_counts())

    # 标签映射，固定顺序
    label2id = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    id2label = {v: k for k, v in label2id.items()}

    sampled_df["labels"] = sampled_df[label_col].map(label2id)

    train_df, test_df = train_test_split(
        sampled_df,
        test_size=0.2,
        random_state=seed,
        stratify=sampled_df["labels"]
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=seed,
        stratify=train_df["labels"]
    )

    print("\nTrain size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))

    return train_df, val_df, test_df, label2id, id2label


# =========================
# 4. Tokenization
# =========================
def tokenize_data(train_df, val_df, test_df, tokenizer, text_col: str, max_length: int):
    def tokenize_function(batch):
        return tokenizer(
            batch[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_dataset = Dataset.from_pandas(train_df[[text_col, "labels"]], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[[text_col, "labels"]], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[[text_col, "labels"]], preserve_index=False)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    cols_to_remove = [text_col]
    train_dataset = train_dataset.remove_columns(cols_to_remove)
    val_dataset = val_dataset.remove_columns(cols_to_remove)
    test_dataset = test_dataset.remove_columns(cols_to_remove)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, val_dataset, test_dataset


# =========================
# 5. 评价指标
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# =========================
# 6. 兼容不同 transformers 版本的 TrainingArguments
# =========================
def build_training_args(
    output_dir,
    learning_rate,
    train_batch_size,
    eval_batch_size,
    num_epochs,
    weight_decay,
    fp16,
):
    args_signature = inspect.signature(TrainingArguments.__init__)
    args_params = args_signature.parameters

    strategy_key = "eval_strategy" if "eval_strategy" in args_params else "evaluation_strategy"

    kwargs = {
        "output_dir": output_dir,
        strategy_key: "epoch",
        "save_strategy": "epoch",
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "num_train_epochs": num_epochs,
        "weight_decay": weight_decay,
        "logging_dir": os.path.join(output_dir, "logs"),
        "logging_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": fp16,
    }

    return TrainingArguments(**kwargs)


# =========================
# 7. 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/clean_reviews.csv",
        help="Path to cleaned review dataset."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained transformer model name."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/transformer_sentiment",
        help="Directory to save trained model."
    )

    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="outputs",
        help="Directory to save metrics and classification report."
    )

    parser.add_argument(
        "--max_per_class",
        type=int,
        default=10000,
        help="Maximum number of samples per sentiment class."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token length."
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=2,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Training batch size per device."
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size per device."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate."
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()

    set_seed(args.seed)

    print("\n========== Device Check ==========")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    fp16 = torch.cuda.is_available()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, label2id, id2label = load_and_prepare_data(
        data_path=args.data_path,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    print("\n========== Loading tokenizer and model ==========")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset, val_dataset, test_dataset = tokenize_data(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        text_col="review_text",
        max_length=args.max_length,
    )

    # 计算类别权重，提升少数类 neutral 的重要性
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["labels"].values,
    )

    print("\nClass weights:")
    print({
        id2label[i]: float(class_weights[i])
        for i in range(len(class_weights))
    })

    training_args = build_training_args(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        fp16=fp16,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("\n========== Start training ==========")
    trainer.train()

    print("\n========== Evaluate on test set ==========")
    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    target_names = [id2label[i] for i in range(3)]

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=target_names,
        digits=4,
    )

    print("\nClassification Report:")
    print(report)

    # 保存模型和 tokenizer
    print("\n========== Saving model ==========")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 保存 label mapping
    label_mapping_path = os.path.join(args.output_dir, "label_mapping.json")
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 保存 metrics
    metrics_path = os.path.join(args.metrics_dir, "transformer_sentiment_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    # 保存 classification report
    report_path = os.path.join(args.metrics_dir, "transformer_sentiment_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved model to:", args.output_dir)
    print("Saved metrics to:", metrics_path)
    print("Saved report to:", report_path)

    print("\n========== Baseline comparison reference ==========")
    print("Previous TF-IDF + Logistic Regression baseline:")
    print("Accuracy = 0.7890")
    print("Macro-F1 = 0.6660")


if __name__ == "__main__":
    main()