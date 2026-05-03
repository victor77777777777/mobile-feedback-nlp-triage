import argparse
from pathlib import Path

import pandas as pd


ISSUE_LABELS = [
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
ISSUE_KEYWORDS = {
    "battery": [
        "battery", "charge", "charging", "charger", "power", "drain",
        "dies", "dead", "hot", "overheat", "overheating"
    ],
    "screen": [
        "screen", "display", "touch", "touchscreen", "crack", "cracked",
        "pixel", "brightness"
    ],
    "performance": [
        "slow", "lag", "laggy", "freeze", "freezes", "frozen", "crash",
        "crashes", "stuck", "hang", "unresponsive", "performance"
    ],
    "network": [
        "signal", "wifi", "wi-fi", "bluetooth", "network", "sim",
        "call", "calls", "data", "lte", "4g", "3g"
    ],
    "camera": [
        "camera", "photo", "photos", "picture", "pictures", "video",
        "focus", "blurry", "blur"
    ],
    "shipping": [
        "shipping", "delivery", "delivered", "arrived", "package",
        "packaging", "box", "late", "missing"
    ],
    "service": [
        "service", "support", "seller", "warranty", "refund", "return",
        "replacement", "repair", "customer"
    ],
    "price": [
        "price", "value", "money", "cheap", "expensive", "deal", "worth", "cost"
    ],
    "condition": [
        "used", "refurbished", "scratch", "scratches", "dented", "dent",
        "condition", "new", "old"
    ],
    "software": [
        "software", "update", "android", "system", "os", "app", "apps",
        "ui", "bug"
    ],
}


def infer_issue_label(text: str) -> str:
    text = str(text).lower()

    scores = {}

    for label, keywords in ISSUE_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 1
        scores[label] = score

    best_label = max(scores, key=scores.get)

    if scores[best_label] == 0:
        return "other"

    return best_label


def stratified_sample_by_issue(sub_df: pd.DataFrame, target_n: int, seed: int = 42) -> pd.DataFrame:
    if len(sub_df) == 0 or target_n <= 0:
        return sub_df.iloc[0:0].copy()

    if len(sub_df) <= target_n:
        return sub_df.copy()

    sampled_parts = []
    issue_labels = sub_df["rule_issue_label"].value_counts().index.tolist()

    per_issue_n = max(1, target_n // max(1, len(issue_labels)))
    used_indices = set()

    for issue in issue_labels:
        issue_df = sub_df[sub_df["rule_issue_label"] == issue]
        take_n = min(per_issue_n, len(issue_df))

        part = issue_df.sample(n=take_n, random_state=seed)
        sampled_parts.append(part)
        used_indices.update(part.index.tolist())

    sampled_df = pd.concat(sampled_parts, axis=0)

    if len(sampled_df) < target_n:
        remaining_df = sub_df.drop(index=list(used_indices), errors="ignore")

        if len(remaining_df) > 0:
            extra_n = min(target_n - len(sampled_df), len(remaining_df))
            extra_df = remaining_df.sample(n=extra_n, random_state=seed)
            sampled_df = pd.concat([sampled_df, extra_df], axis=0)

    if len(sampled_df) > target_n:
        sampled_df = sampled_df.sample(n=target_n, random_state=seed)

    return sampled_df.copy()


def build_annotation_sample(
    input_path: str,
    output_path: str,
    total_size: int = 2000,
    seed: int = 42,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    print("\n========== 读取清洗后的评论数据 ==========")
    df = pd.read_csv(input_path)

    required_cols = [
        "review_id",
        "review_text",
        "rating",
        "sentiment_label",
        "issue_label",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入数据缺少必要列：{col}")

    df = df[required_cols].copy()

    df = df.dropna(subset=["review_text", "sentiment_label"])

    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""]

    df["issue_label"] = df["issue_label"].fillna("")
    df["issue_label"] = df["issue_label"].astype(str).str.strip().str.lower()

    missing_issue_mask = (df["issue_label"] == "") | (df["issue_label"] == "nan")

    df.loc[missing_issue_mask, "issue_label"] = df.loc[
        missing_issue_mask,
        "review_text"
    ].apply(infer_issue_label)

    df = df.rename(columns={"issue_label": "rule_issue_label"})

    valid_sentiments = ["negative", "neutral", "positive"]
    df = df[df["sentiment_label"].isin(valid_sentiments)]

    print(f"可用数据量：{len(df)}")

    print("\n原始情感分布：")
    print(df["sentiment_label"].value_counts())

    print("\n原始规则 issue 分布：")
    print(df["rule_issue_label"].value_counts())

    n_negative = int(total_size * 0.55)
    n_neutral = int(total_size * 0.25)
    n_positive = total_size - n_negative - n_neutral

    negative_df = df[df["sentiment_label"] == "negative"]
    neutral_df = df[df["sentiment_label"] == "neutral"]
    positive_df = df[df["sentiment_label"] == "positive"]

    negative_sample = stratified_sample_by_issue(
        negative_df,
        target_n=n_negative,
        seed=seed,
    )

    neutral_sample = stratified_sample_by_issue(
        neutral_df,
        target_n=n_neutral,
        seed=seed,
    )

    positive_sample = stratified_sample_by_issue(
        positive_df,
        target_n=n_positive,
        seed=seed,
    )

    sample_df = pd.concat(
        [
            negative_sample,
            neutral_sample,
            positive_sample,
        ],
        axis=0,
    )

    if len(sample_df) < total_size:
        used_review_ids = set(sample_df["review_id"].tolist())
        remaining_df = df[~df["review_id"].isin(used_review_ids)]

        if len(remaining_df) > 0:
            extra_n = min(total_size - len(sample_df), len(remaining_df))
            extra_df = remaining_df.sample(n=extra_n, random_state=seed)
            sample_df = pd.concat([sample_df, extra_df], axis=0)

    sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    sample_df.insert(0, "annotation_id", range(1, len(sample_df) + 1))

    sample_df["manual_issue_label"] = ""
    sample_df["annotation_note"] = ""

    sample_df = sample_df[
        [
            "annotation_id",
            "review_id",
            "review_text",
            "rating",
            "sentiment_label",
            "rule_issue_label",
            "manual_issue_label",
            "annotation_note",
        ]
    ]

    sample_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== 人工标注样本生成完成 ==========")
    print(f"输出路径：{output_path}")
    print(f"总行数：{len(sample_df)}")

    print("\n样本情感分布：")
    print(sample_df["sentiment_label"].value_counts())

    print("\n样本规则 issue 分布：")
    print(sample_df["rule_issue_label"].value_counts())

    print("\nmanual_issue_label 可填写的标签：")
    for label in ISSUE_LABELS:
        print(f"- {label}")


def create_labeling_guideline(output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    guideline = """# Issue 人工标注说明文档

`rule_issue_label` 是旧的关键词规则标签，只能作为参考。
最终用于训练模型的标签需要填写在 `manual_issue_label` 这一列。

## 可用标签

battery
screen
performance
network
camera
shipping
service
price
condition
software
other

## 标注规则

1. 只选择评论中最主要的问题。
2. 如果一条评论提到多个问题，选择最严重或最强调的问题。
3. 如果是正面评论，但明确提到某个功能，也可以标成对应功能。
4. 如果评论太模糊，或者没有明确 issue，标成 other。
5. 不要直接照抄 rule_issue_label，它只是参考。
6. 标签必须全部小写。
7. 不要创造新的标签。
"""

    output_path.write_text(guideline, encoding="utf-8")

    print(f"\n标注说明文档已保存到：{output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/clean_reviews.csv",
        help="清洗后评论数据路径",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/annotated/issue_annotation_sample.csv",
        help="人工标注样本输出路径",
    )

    parser.add_argument(
        "--guideline_path",
        type=str,
        default="docs/issue_labeling_guideline.md",
        help="issue 标注说明文档输出路径",
    )

    parser.add_argument(
        "--total_size",
        type=int,
        default=2000,
        help="抽样总数量",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    args = parser.parse_args()

    build_annotation_sample(
        input_path=args.input_path,
        output_path=args.output_path,
        total_size=args.total_size,
        seed=args.seed,
    )

    create_labeling_guideline(
        output_path=args.guideline_path,
    )


if __name__ == "__main__":
    main()