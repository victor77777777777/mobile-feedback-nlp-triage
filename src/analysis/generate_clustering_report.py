import argparse
from pathlib import Path

import pandas as pd


def clean_text(text: str, max_len: int = 500) -> str:
    if pd.isna(text):
        return ""

    text = str(text).replace("\n", " ").replace("\r", " ").strip()

    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."

    return text


def generate_clustering_report(
    input_path: str,
    output_path: str,
    top_n_clusters: int = 20,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_cols = [
        "cluster_id",
        "cluster_theme",
        "cluster_size",
        "top_keywords",
        "major_issue",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入文件缺少必要列：{col}")

    df = df.sort_values(by="cluster_size", ascending=False).reset_index(drop=True)

    total_reviews = int(df["cluster_size"].sum())
    total_clusters = len(df)

    theme_counts = df["cluster_theme"].value_counts()

    lines = []

    lines.append("# Review Clustering and Topic Discovery Report")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(
        "This report summarizes recurring complaint themes discovered from customer reviews "
        "using sentence embeddings and MiniBatchKMeans clustering."
    )
    lines.append("")
    lines.append(f"- Total clustered reviews: **{total_reviews}**")
    lines.append(f"- Number of clusters: **{total_clusters}**")
    lines.append("- Input reviews: negative customer reviews")
    lines.append("- Embedding method: sentence-transformers/all-MiniLM-L6-v2")
    lines.append("- Clustering method: MiniBatchKMeans")
    lines.append("")

    if "silhouette_score_global" in df.columns:
        silhouette_values = df["silhouette_score_global"].dropna().unique()
        if len(silhouette_values) > 0:
            lines.append(f"- Global silhouette score: **{float(silhouette_values[0]):.4f}**")
            lines.append("")

    lines.append("## 2. Cluster Theme Distribution")
    lines.append("")
    lines.append("| Cluster Theme | Number of Clusters |")
    lines.append("|---|---:|")

    for theme, count in theme_counts.items():
        lines.append(f"| {theme} | {count} |")

    lines.append("")

    lines.append("## 3. Top Clusters by Size")
    lines.append("")
    lines.append("| Cluster ID | Theme | Size | Major Issue | Keywords |")
    lines.append("|---:|---|---:|---|---|")

    for _, row in df.head(top_n_clusters).iterrows():
        cluster_id = row["cluster_id"]
        theme = clean_text(row["cluster_theme"], 120)
        size = int(row["cluster_size"])
        major_issue = clean_text(row["major_issue"], 80)
        keywords = clean_text(row["top_keywords"], 200)

        lines.append(
            f"| {cluster_id} | {theme} | {size} | {major_issue} | {keywords} |"
        )

    lines.append("")

    lines.append("## 4. Detailed Cluster Summaries")
    lines.append("")

    for _, row in df.head(top_n_clusters).iterrows():
        cluster_id = row["cluster_id"]
        theme = clean_text(row["cluster_theme"], 200)
        size = int(row["cluster_size"])
        major_issue = clean_text(row["major_issue"], 100)
        keywords = clean_text(row["top_keywords"], 300)

        lines.append(f"### Cluster {cluster_id}: {theme}")
        lines.append("")
        lines.append(f"- Cluster size: **{size}**")
        lines.append(f"- Major issue label: **{major_issue}**")
        lines.append(f"- Top keywords: {keywords}")

        if "major_issue_count" in row:
            try:
                lines.append(f"- Major issue count: **{int(row['major_issue_count'])}**")
            except Exception:
                pass

        if "avg_issue_confidence" in row:
            try:
                lines.append(
                    f"- Average issue confidence: **{float(row['avg_issue_confidence']):.4f}**"
                )
            except Exception:
                pass

        if "negative_ratio" in row:
            try:
                lines.append(f"- Negative ratio: **{float(row['negative_ratio']):.2f}**")
            except Exception:
                pass

        lines.append("")
        lines.append("Representative reviews:")
        lines.append("")

        for i in range(1, 4):
            col = f"representative_review_{i}"
            if col in df.columns:
                review = clean_text(row.get(col, ""), 700)
                if review:
                    lines.append(f"{i}. {review}")

        lines.append("")

    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(
        "The clustering results reveal several recurring complaint themes in negative reviews. "
        "Common themes include battery and charging failures, carrier or SIM compatibility problems, "
        "screen and display defects, refurbished or damaged product condition, software and app stability issues, "
        "and service or warranty-related complaints."
    )
    lines.append("")
    lines.append(
        "These clusters complement the supervised issue classifier. While the issue classifier assigns predefined "
        "categories to individual reviews, clustering helps discover repeated sub-themes and representative examples "
        "within the review corpus."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n========== 聚类报告生成完成 ==========")
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")
    print(f"总评论数：{total_reviews}")
    print(f"Cluster 数量：{total_clusters}")
    print("\n主题分布：")
    print(theme_counts)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/clustering/cluster_summary_labeled.csv",
        help="带主题名的 cluster summary 文件路径",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/clustering/clustering_report.md",
        help="聚类报告输出路径",
    )

    parser.add_argument(
        "--top_n_clusters",
        type=int,
        default=20,
        help="报告中展示的 cluster 数量",
    )

    args = parser.parse_args()

    generate_clustering_report(
        input_path=args.input_path,
        output_path=args.output_path,
        top_n_clusters=args.top_n_clusters,
    )


if __name__ == "__main__":
    main()