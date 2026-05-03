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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


VALID_ISSUE_LABELS = [
    "battery",
    "screen",
    "performance",
    "network",
    "camera",
    "shipping",
    "service",
    "price",
    "condition",
    "software",
    "other",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights,
                dtype=torch.float,
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
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


def load_and_prepare_data(
    data_path: str,
    text_col: str = "review_text",
    label_col: str = "manual_issue_label",
    seed: int = 42,
):
    print("\n========== Loading issue annotation data ==========")

    df = pd.read_csv(data_path)

    required_cols = [text_col, label_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    df = df[[text_col, label_col]].copy()

    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    df = df[df[text_col] != ""]
    df = df[df[label_col] != ""]
    df = df[df[label_col] != "nan"]

    df = df[df[label_col].isin(VALID_ISSUE_LABELS)]

    print(f"Usable labeled rows: {len(df)}")

    print("\nIssue label distribution:")
    print(df[label_col].value_counts())

    label2id = {
        label: idx for idx, label in enumerate(VALID_ISSUE_LABELS)
    }

    id2label = {
        idx: label for label, idx in label2id.items()
    }

    df["labels"] = df[label_col].map(label2id)

    label_counts = df["labels"].value_counts()

    rare_labels = label_counts[label_counts < 2]

    if len(rare_labels) > 0:
        print("\nWarning: Some labels have fewer than 2 samples:")
        for label_id, count in rare_labels.items():
            print(f"{id2label[int(label_id)]}: {count}")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["labels"],
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=seed,
        stratify=train_df["labels"],
    )

    print("\nTrain size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))

    print("\nTrain label distribution:")
    print(train_df[label_col].value_counts())

    print("\nValidation label distribution:")
    print(val_df[label_col].value_counts())

    print("\nTest label distribution:")
    print(test_df[label_col].value_counts())

    return train_df, val_df, test_df, label2id, id2label


def tokenize_data(
    train_df,
    val_df,
    test_df,
    tokenizer,
    text_col: str,
    max_length: int,
):
    def tokenize_function(batch):
        return tokenizer(
            batch[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_dataset = Dataset.from_pandas(
        train_df[[text_col, "labels"]],
        preserve_index=False,
    )

    val_dataset = Dataset.from_pandas(
        val_df[[text_col, "labels"]],
        preserve_index=False,
    )

    test_dataset = Dataset.from_pandas(
        test_df[[text_col, "labels"]],
        preserve_index=False,
    )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns([text_col])
    val_dataset = val_dataset.remove_columns([text_col])
    test_dataset = test_dataset.remove_columns([text_col])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, val_dataset, test_dataset


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
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": fp16,
    }

    return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/annotated/issue_annotation_bootstrap_final_reviewed.csv",
        help="Path to issue annotation CSV.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained transformer model name.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/transformer_issue",
        help="Directory to save trained issue classifier.",
    )

    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="outputs",
        help="Directory to save metrics and reports.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=160,
        help="Maximum token length.",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=5,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Training batch size per device.",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size per device.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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
        seed=args.seed,
    )

    print("\n========== Loading tokenizer and model ==========")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(VALID_ISSUE_LABELS),
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

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(VALID_ISSUE_LABELS)),
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

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
        "class_weights": class_weights,
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    trainer_params = trainer_signature.parameters

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    print("\n========== Start issue classifier training ==========")
    trainer.train()

    print("\n========== Evaluate on test set ==========")

    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    pred_labels = np.argmax(logits, axis=-1)
    true_labels = predictions.label_ids

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=-1,
    ).numpy()

    target_names = [id2label[i] for i in range(len(VALID_ISSUE_LABELS))]

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    # 保存测试集预测明细，方便错误分析
    test_analysis_df = test_df.copy().reset_index(drop=True)

    test_analysis_df["true_label"] = [
        id2label[int(label_id)] for label_id in true_labels
    ]

    test_analysis_df["pred_label"] = [
        id2label[int(label_id)] for label_id in pred_labels
    ]

    test_analysis_df["correct"] = (
            test_analysis_df["true_label"] == test_analysis_df["pred_label"]
    )

    test_analysis_df["confidence"] = probabilities.max(axis=1)

    top_k = min(3, probabilities.shape[1])
    top_indices = np.argsort(-probabilities, axis=1)[:, :top_k]

    for k in range(top_k):
        test_analysis_df[f"top_{k + 1}_label"] = [
            id2label[int(idx)] for idx in top_indices[:, k]
        ]
        test_analysis_df[f"top_{k + 1}_prob"] = [
            float(probabilities[row_idx, top_indices[row_idx, k]])
            for row_idx in range(len(test_analysis_df))
        ]

    prediction_path = os.path.join(
        args.metrics_dir,
        "transformer_issue_test_predictions.csv",
    )

    test_analysis_df.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 保存混淆矩阵
    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=list(range(len(VALID_ISSUE_LABELS))),
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in target_names],
        columns=[f"pred_{label}" for label in target_names],
    )

    confusion_matrix_path = os.path.join(
        args.metrics_dir,
        "transformer_issue_confusion_matrix.csv",
    )

    cm_df.to_csv(
        confusion_matrix_path,
        encoding="utf-8-sig",
    )

    print("\nClassification Report:")
    print(report)

    print("\n========== Saving model ==========")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    label_mapping_path = os.path.join(args.output_dir, "label_mapping.json")

    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
                "valid_issue_labels": VALID_ISSUE_LABELS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    metrics_path = os.path.join(args.metrics_dir, "transformer_issue_metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    report_path = os.path.join(args.metrics_dir, "transformer_issue_classification_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved model to:", args.output_dir)
    print("Saved metrics to:", metrics_path)
    print("Saved report to:", report_path)
    print("Saved test predictions to:", prediction_path)
    print("Saved confusion matrix to:", confusion_matrix_path)

if __name__ == "__main__":
    main()