import argparse
from pathlib import Path

import pandas as pd


def min_max_normalize(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series([1.0] * len(series), index=series.index)

    return (series - min_value) / (max_value - min_value)


def clean_text(value, max_len: int = 500) -> str:
    if pd.isna(value):
        return ""

    text = str(value).replace("\n", " ").replace("\r", " ").strip()

    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."

    return text


def calculate_priority_scores(
    input_path: str,
    output_path: str,
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
        "major_issue",
        "top_keywords",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入文件缺少必要列：{col}")

    if "negative_ratio" not in df.columns:
        df["negative_ratio"] = 1.0

    if "avg_issue_confidence" not in df.columns:
        df["avg_issue_confidence"] = 0.5

    if "major_issue_count" not in df.columns:
        df["major_issue_count"] = df["cluster_size"]

    df["cluster_size"] = pd.to_numeric(df["cluster_size"], errors="coerce").fillna(0)
    df["negative_ratio"] = pd.to_numeric(df["negative_ratio"], errors="coerce").fillna(1.0)
    df["avg_issue_confidence"] = pd.to_numeric(df["avg_issue_confidence"], errors="coerce").fillna(0.5)
    df["major_issue_count"] = pd.to_numeric(df["major_issue_count"], errors="coerce").fillna(0)

    # 1. 规模分数：cluster 越大，优先级越高
    df["cluster_size_score"] = min_max_normalize(df["cluster_size"])

    # 2. issue 集中度分数：某个 issue 在 cluster 内越集中，说明主题越清晰
    df["issue_concentration"] = df["major_issue_count"] / df["cluster_size"].replace(0, 1)
    df["issue_concentration_score"] = min_max_normalize(df["issue_concentration"])

    # 3. 负面程度分数：当前我们聚类的是 negative reviews，所以通常为 1
    df["negative_ratio_score"] = min_max_normalize(df["negative_ratio"])

    # 4. 模型置信度分数
    df["confidence_score"] = min_max_normalize(df["avg_issue_confidence"])

    # 当前数据没有可靠 date，所以暂时无法计算 growth rate
    # 这里用 0.5 作为 neutral growth proxy，并在报告中说明这是未来扩展项
    df["growth_proxy_score"] = 0.5

    # Priority score:
    # 规模越大、主题越集中、负面比例越高、模型越确信，优先级越高
    df["priority_score"] = (
        0.40 * df["cluster_size_score"]
        + 0.25 * df["issue_concentration_score"]
        + 0.20 * df["confidence_score"]
        + 0.10 * df["negative_ratio_score"]
        + 0.05 * df["growth_proxy_score"]
    )

    df["priority_score"] = (df["priority_score"] * 100).round(2)

    df = df.sort_values(
        by="priority_score",
        ascending=False,
    ).reset_index(drop=True)

    df.insert(0, "priority_rank", range(1, len(df) + 1))

    for i in range(1, 4):
        col = f"representative_review_{i}"
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_text(x, max_len=600))

    output_cols = [
        "priority_rank",
        "cluster_id",
        "cluster_theme",
        "priority_score",
        "cluster_size",
        "major_issue",
        "major_issue_count",
        "issue_concentration",
        "avg_issue_confidence",
        "negative_ratio",
        "top_keywords",
        "representative_review_1",
        "representative_review_2",
        "representative_review_3",
        "cluster_size_score",
        "issue_concentration_score",
        "confidence_score",
        "negative_ratio_score",
        "growth_proxy_score",
    ]

    output_cols = [col for col in output_cols if col in df.columns]
    df = df[output_cols]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== Priority scoring 完成 ==========")
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")
    print(f"Cluster 数量：{len(df)}")

    print("\nTop priority issues:")
    display_cols = [
        "priority_rank",
        "cluster_id",
        "cluster_theme",
        "priority_score",
        "cluster_size",
        "major_issue",
        "top_keywords",
    ]

    print(df[display_cols].head(10))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/clustering/cluster_summary_labeled.csv",
        help="带主题标签的 cluster summary 文件路径",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/priority/priority_scores.csv",
        help="priority score 输出路径",
    )

    args = parser.parse_args()

    calculate_priority_scores(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()