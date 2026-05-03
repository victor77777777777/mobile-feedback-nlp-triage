import argparse
from pathlib import Path

import pandas as pd


VALID_ISSUE_LABELS = {
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
}


HIGH_REVIEW_KEYWORDS = {
    "performance": [
        "slow",
        "lag",
        "laggy",
        "freeze",
        "freezes",
        "frozen",
        "crash",
        "crashes",
        "stuck",
        "hang",
        "unresponsive",
    ],
    "battery": [
        "battery",
        "charge",
        "charging",
        "charger",
        "power",
        "drain",
        "dies",
        "dead",
        "hot",
        "overheat",
        "overheating",
    ],
    "screen": [
        "screen",
        "display",
        "touch",
        "touchscreen",
        "crack",
        "cracked",
        "pixel",
        "brightness",
    ],
    "network": [
        "signal",
        "wifi",
        "wi-fi",
        "bluetooth",
        "network",
        "sim",
        "call",
        "calls",
        "data",
        "lte",
        "4g",
        "3g",
    ],
    "camera": [
        "camera",
        "photo",
        "photos",
        "picture",
        "pictures",
        "video",
        "focus",
        "blurry",
        "blur",
    ],
    "shipping": [
        "shipping",
        "delivery",
        "delivered",
        "arrived",
        "package",
        "packaging",
        "box",
        "late",
        "missing",
    ],
    "service": [
        "service",
        "support",
        "seller",
        "warranty",
        "refund",
        "return",
        "replacement",
        "repair",
        "customer",
    ],
    "price": [
        "price",
        "value",
        "money",
        "cheap",
        "expensive",
        "deal",
        "worth",
        "cost",
    ],
    "condition": [
        "used",
        "refurbished",
        "scratch",
        "scratches",
        "dented",
        "dent",
        "condition",
        "new",
        "old",
    ],
    "software": [
        "software",
        "update",
        "android",
        "system",
        "os",
        "app",
        "apps",
        "ui",
        "bug",
    ],
}


def count_keyword_matches(text: str, keywords: list[str]) -> int:
    text = str(text).lower()
    count = 0

    for keyword in keywords:
        if keyword in text:
            count += 1

    return count


def guess_issue_by_keywords(text: str) -> tuple[str, int]:
    scores = {}

    for label, keywords in HIGH_REVIEW_KEYWORDS.items():
        scores[label] = count_keyword_matches(text, keywords)

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score <= 0:
        return "other", 0

    return best_label, best_score


def bootstrap_issue_labels(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_cols = [
        "annotation_id",
        "review_id",
        "review_text",
        "rating",
        "sentiment_label",
        "rule_issue_label",
        "manual_issue_label",
        "annotation_note",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入文件缺少必要列：{col}")

    df["review_text"] = df["review_text"].astype(str)
    df["rule_issue_label"] = df["rule_issue_label"].astype(str).str.strip().str.lower()

    df["manual_issue_label"] = df["rule_issue_label"]

    df.loc[
        ~df["manual_issue_label"].isin(VALID_ISSUE_LABELS),
        "manual_issue_label",
    ] = "other"

    guessed_labels = []
    guessed_scores = []

    for text in df["review_text"]:
        guessed_label, guessed_score = guess_issue_by_keywords(text)
        guessed_labels.append(guessed_label)
        guessed_scores.append(guessed_score)

    df["keyword_guess_label"] = guessed_labels
    df["keyword_guess_score"] = guessed_scores

    df["needs_review"] = "no"
    df["review_reason"] = ""

    mask_other = df["manual_issue_label"] == "other"
    df.loc[mask_other, "needs_review"] = "yes"
    df.loc[mask_other, "review_reason"] = "规则标签为 other，建议人工确认"

    mask_conflict = (
        (df["keyword_guess_score"] > 0)
        & (df["keyword_guess_label"] != df["manual_issue_label"])
    )
    df.loc[mask_conflict, "needs_review"] = "yes"
    df.loc[mask_conflict, "review_reason"] = "关键词辅助判断与规则标签不一致"

    mask_negative_positive_issue = (
        (df["sentiment_label"] == "negative")
        & (df["manual_issue_label"].isin(["price", "other"]))
    )
    df.loc[mask_negative_positive_issue, "needs_review"] = "yes"
    df.loc[mask_negative_positive_issue, "review_reason"] = "负面评论但标签为 price/other，建议人工确认"

    mask_positive_complaint_issue = (
        (df["sentiment_label"] == "positive")
        & (df["manual_issue_label"].isin(["battery", "screen", "performance", "network", "service"]))
    )
    df.loc[mask_positive_complaint_issue, "needs_review"] = "yes"
    df.loc[mask_positive_complaint_issue, "review_reason"] = "正面评论但标签为投诉类 issue，建议人工确认"

    output_cols = [
        "annotation_id",
        "review_id",
        "review_text",
        "rating",
        "sentiment_label",
        "rule_issue_label",
        "keyword_guess_label",
        "keyword_guess_score",
        "manual_issue_label",
        "needs_review",
        "review_reason",
        "annotation_note",
    ]

    df = df[output_cols]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== 半自动 issue 标注文件生成完成 ==========")
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")
    print(f"总行数：{len(df)}")

    print("\nmanual_issue_label 分布：")
    print(df["manual_issue_label"].value_counts())

    print("\nneeds_review 分布：")
    print(df["needs_review"].value_counts())

    print("\n需要人工检查的主要原因：")
    print(df[df["needs_review"] == "yes"]["review_reason"].value_counts())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/annotated/issue_annotation_sample.csv",
        help="人工标注样本输入路径",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/annotated/issue_annotation_bootstrap.csv",
        help="半自动标注输出路径",
    )

    args = parser.parse_args()

    bootstrap_issue_labels(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()