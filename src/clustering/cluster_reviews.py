import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_ISSUE_ID2LABEL = {
    0: "battery",
    1: "screen",
    2: "performance",
    3: "network",
    4: "camera",
    5: "shipping",
    6: "service",
    7: "price",
    8: "condition",
    9: "software",
    10: "other",
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_issue_label_mapping(model_dir: str) -> dict[int, str]:
    mapping_path = Path(model_dir) / "label_mapping.json"

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        id2label = mapping.get("id2label", DEFAULT_ISSUE_ID2LABEL)
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        id2label = DEFAULT_ISSUE_ID2LABEL

    return id2label


def load_review_data(
    input_path: str,
    sentiment_filter: str = "negative",
    max_samples: int = 20000,
    seed: int = 42,
):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    print("\n========== 读取评论数据 ==========")
    df = pd.read_csv(input_path)

    required_cols = [
        "review_id",
        "review_text",
        "sentiment_label",
        "rating",
        "product_name",
        "brand",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入数据缺少必要列：{col}")

    df = df[required_cols].copy()
    df = df.dropna(subset=["review_text", "sentiment_label"])

    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""]

    if sentiment_filter.lower() != "all":
        df = df[df["sentiment_label"] == sentiment_filter]

    print(f"筛选后数据量：{len(df)}")

    if len(df) == 0:
        raise ValueError("筛选后没有可用评论，请检查 sentiment_filter。")

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    print(f"实际用于聚类的数据量：{len(df)}")

    return df


def predict_issue_labels(
    df: pd.DataFrame,
    issue_model_dir: str,
    max_length: int = 256,
    batch_size: int = 32,
):
    if not os.path.exists(issue_model_dir):
        raise FileNotFoundError(
            f"找不到 issue 模型目录：{issue_model_dir}\n"
            "请先训练 issue classifier。"
        )

    print("\n========== 使用 Transformer issue model 预测 issue 标签 ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(issue_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(issue_model_dir)

    model.to(device)
    model.eval()

    id2label = load_issue_label_mapping(issue_model_dir)

    texts = df["review_text"].tolist()

    predicted_labels = []
    predicted_confidences = []

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)

        pred_ids = torch.argmax(probs, dim=-1)
        pred_confs = torch.max(probs, dim=-1).values

        for pred_id, conf in zip(pred_ids, pred_confs):
            label_id = int(pred_id.item())
            predicted_labels.append(id2label[label_id])
            predicted_confidences.append(float(conf.item()))

        if (start_idx // batch_size + 1) % 20 == 0:
            print(f"已预测：{min(start_idx + batch_size, len(texts))}/{len(texts)}")

    df["predicted_issue_label"] = predicted_labels
    df["predicted_issue_confidence"] = predicted_confidences

    print("\n预测 issue 分布：")
    print(df["predicted_issue_label"].value_counts())

    return df


def create_sentence_embeddings(
    texts: list[str],
    embedding_model_name: str,
    batch_size: int = 64,
):
    print("\n========== 生成 sentence embeddings ==========")
    print(f"Embedding model: {embedding_model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        embedding_model_name,
        device=device,
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def run_kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int = 20,
    seed: int = 42,
):
    print("\n========== 运行 MiniBatchKMeans 聚类 ==========")
    print(f"n_clusters: {n_clusters}")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=2048,
        n_init="auto",
    )

    cluster_labels = kmeans.fit_predict(embeddings)

    distances = np.linalg.norm(
        embeddings - kmeans.cluster_centers_[cluster_labels],
        axis=1,
    )

    print("聚类完成。")

    if len(set(cluster_labels)) > 1 and len(embeddings) <= 30000:
        print("\n正在计算 silhouette score...")
        score = silhouette_score(
            embeddings,
            cluster_labels,
            metric="cosine",
            sample_size=min(5000, len(embeddings)),
            random_state=seed,
        )
        print(f"Silhouette score: {score:.4f}")
    else:
        score = None

    return cluster_labels, distances, score


def extract_cluster_keywords(
    df: pd.DataFrame,
    text_col: str = "review_text",
    cluster_col: str = "cluster_id",
    top_n: int = 10,
):
    print("\n========== 提取 cluster 关键词 ==========")

    custom_stop_words = [
        "phone", "phones", "cell", "cellphone", "smartphone",
        "product", "item", "device",
        "just", "really", "very", "good", "great", "bad",
        "like", "love", "nice", "excellent", "awesome",
        "buy", "bought", "purchase", "purchased", "ordered",
        "got", "get", "received", "came", "come",
        "work", "works", "worked", "working",
        "use", "used", "using",
        "time", "day", "days", "week", "weeks", "month", "months",
        "thing", "things", "way", "lot", "bit",
        "don", "doesn", "didn", "isn", "wasn", "can", "could",
        "amazon", "seller", "review", "reviews",
        "star", "stars", "money",
        "new", "did", "does", "return", "returned", "returns",
        "came", "problem", "problems", "issue", "issues",
        "waste", "recommend", "recommended",
        "try", "tried", "know", "want", "wanted",
        "going", "make", "made", "right", "away",
        "box", "case"
    ]

    cluster_texts = (
        df.groupby(cluster_col)[text_col]
        .apply(lambda texts: " ".join(texts.astype(str).tolist()))
        .reset_index()
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
    )

    tfidf_matrix = vectorizer.fit_transform(cluster_texts[text_col])
    feature_names = np.array(vectorizer.get_feature_names_out())

    cluster_keywords = {}

    custom_stop_set = set(custom_stop_words)

    for row_idx, cluster_id in enumerate(cluster_texts[cluster_col]):
        row = tfidf_matrix[row_idx].toarray().flatten()
        sorted_indices = row.argsort()[::-1]

        keywords = []

        for idx in sorted_indices:
            term = feature_names[idx]
            term_words = term.split()

            if any(word in custom_stop_set for word in term_words):
                continue

            if len(term) <= 2:
                continue

            keywords.append(term)

            if len(keywords) >= top_n:
                break

        cluster_keywords[cluster_id] = ", ".join(keywords)

    return cluster_keywords
    print("\n========== 提取 cluster 关键词 ==========")

    cluster_texts = (
        df.groupby(cluster_col)[text_col]
        .apply(lambda texts: " ".join(texts.astype(str).tolist()))
        .reset_index()
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
    )

    tfidf_matrix = vectorizer.fit_transform(cluster_texts[text_col])
    feature_names = np.array(vectorizer.get_feature_names_out())

    cluster_keywords = {}

    for row_idx, cluster_id in enumerate(cluster_texts[cluster_col]):
        row = tfidf_matrix[row_idx].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        keywords = feature_names[top_indices].tolist()
        cluster_keywords[cluster_id] = ", ".join(keywords)

    return cluster_keywords


def build_cluster_summary(
    df: pd.DataFrame,
    output_path: str,
    top_keywords_n: int = 8,
    representative_n: int = 3,
    silhouette: float | None = None,
):
    print("\n========== 生成 cluster summary ==========")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_keywords = extract_cluster_keywords(
        df=df,
        top_n=top_keywords_n,
    )

    summaries = []

    for cluster_id, group in df.groupby("cluster_id"):
        group = group.copy()

        cluster_size = len(group)

        sentiment_counts = group["sentiment_label"].value_counts()
        major_sentiment = sentiment_counts.index[0]
        major_sentiment_count = int(sentiment_counts.iloc[0])

        issue_counts = group["predicted_issue_label"].value_counts()
        major_issue = issue_counts.index[0]
        major_issue_count = int(issue_counts.iloc[0])

        avg_issue_confidence = float(group["predicted_issue_confidence"].mean())

        negative_ratio = float((group["sentiment_label"] == "negative").mean())

        representative_group = group.sort_values(
            by="distance_to_cluster_center",
            ascending=True,
        ).head(representative_n)

        representative_reviews = representative_group["review_text"].tolist()

        row = {
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
            "top_keywords": cluster_keywords.get(cluster_id, ""),
            "major_sentiment": major_sentiment,
            "major_sentiment_count": major_sentiment_count,
            "major_issue": major_issue,
            "major_issue_count": major_issue_count,
            "avg_issue_confidence": avg_issue_confidence,
            "negative_ratio": negative_ratio,
            "silhouette_score_global": silhouette,
        }

        for i in range(representative_n):
            key = f"representative_review_{i + 1}"
            row[key] = representative_reviews[i] if i < len(representative_reviews) else ""

        summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(
        by=["cluster_size"],
        ascending=False,
    ).reset_index(drop=True)

    summary_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Cluster summary 已保存到：{output_path}")

    return summary_df


def save_review_clusters(
    df: pd.DataFrame,
    output_path: str,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_cols = [
        "review_id",
        "review_text",
        "rating",
        "sentiment_label",
        "product_name",
        "brand",
        "predicted_issue_label",
        "predicted_issue_confidence",
        "cluster_id",
        "distance_to_cluster_center",
    ]

    output_cols = [col for col in output_cols if col in df.columns]

    df[output_cols].to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Review cluster results 已保存到：{output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/clean_reviews.csv",
        help="清洗后的评论数据路径",
    )

    parser.add_argument(
        "--issue_model_dir",
        type=str,
        default="models/transformer_issue",
        help="训练好的 issue classifier 模型路径",
    )

    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence embedding 模型名称",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/clustering",
        help="聚类结果输出目录",
    )

    parser.add_argument(
        "--sentiment_filter",
        type=str,
        default="negative",
        choices=["negative", "neutral", "positive", "all"],
        help="选择用于聚类的评论情感类别",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=20000,
        help="最多用于聚类的评论数",
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=20,
        help="KMeans 聚类数量",
    )

    parser.add_argument(
        "--issue_max_length",
        type=int,
        default=256,
        help="issue model 最大 token 长度",
    )

    parser.add_argument(
        "--issue_batch_size",
        type=int,
        default=32,
        help="issue model 推理 batch size",
    )

    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64,
        help="embedding 生成 batch size",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    print("\n========== Device Check ==========")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    review_clusters_path = output_dir / "review_clusters.csv"
    cluster_summary_path = output_dir / "cluster_summary.csv"

    df = load_review_data(
        input_path=args.input_path,
        sentiment_filter=args.sentiment_filter,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    df = predict_issue_labels(
        df=df,
        issue_model_dir=args.issue_model_dir,
        max_length=args.issue_max_length,
        batch_size=args.issue_batch_size,
    )

    embeddings = create_sentence_embeddings(
        texts=df["review_text"].tolist(),
        embedding_model_name=args.embedding_model_name,
        batch_size=args.embedding_batch_size,
    )

    cluster_labels, distances, silhouette = run_kmeans_clustering(
        embeddings=embeddings,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )

    df["cluster_id"] = cluster_labels
    df["distance_to_cluster_center"] = distances

    save_review_clusters(
        df=df,
        output_path=str(review_clusters_path),
    )

    summary_df = build_cluster_summary(
        df=df,
        output_path=str(cluster_summary_path),
        top_keywords_n=8,
        representative_n=3,
        silhouette=silhouette,
    )

    print("\n========== 聚类完成 ==========")
    print(f"每条评论聚类结果：{review_clusters_path}")
    print(f"Cluster 汇总结果：{cluster_summary_path}")

    print("\nTop clusters:")
    print(
        summary_df[
            [
                "cluster_id",
                "cluster_size",
                "top_keywords",
                "major_issue",
                "negative_ratio",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
