import argparse
from pathlib import Path

import pandas as pd


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


PRIORITY_TRUE_LABELS = [
    "condition",
    "performance",
    "shipping",
    "service",
]


def create_error_review_candidates(
    input_path: str,
    output_path: str,
    max_rows: int = 150,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_cols = [
        "review_text",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
        "top_1_label",
        "top_1_prob",
        "top_2_label",
        "top_2_prob",
        "top_3_label",
        "top_3_prob",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入文件缺少必要列：{col}")

    df["true_label"] = df["true_label"].astype(str).str.strip().str.lower()
    df["pred_label"] = df["pred_label"].astype(str).str.strip().str.lower()

    if df["correct"].dtype != bool:
        df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1", "yes"])

    error_df = df[df["correct"] == False].copy()

    priority_df = error_df[
        error_df["true_label"].isin(PRIORITY_TRUE_LABELS)
    ].copy()

    other_error_df = error_df[
        ~error_df["true_label"].isin(PRIORITY_TRUE_LABELS)
    ].copy()

    priority_df["review_priority"] = 1
    other_error_df["review_priority"] = 2

    combined_df = pd.concat(
        [priority_df, other_error_df],
        axis=0,
        ignore_index=True,
    )

    combined_df = combined_df.sort_values(
        by=["review_priority", "true_label", "confidence"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    if len(combined_df) > max_rows:
        combined_df = combined_df.head(max_rows).copy()

    combined_df.insert(0, "review_id_for_error_analysis", range(1, len(combined_df) + 1))

    combined_df["corrected_label"] = combined_df["true_label"]
    combined_df["review_decision"] = ""
    combined_df["review_note"] = ""

    output_cols = [
        "review_id_for_error_analysis",
        "review_text",
        "true_label",
        "pred_label",
        "confidence",
        "top_1_label",
        "top_1_prob",
        "top_2_label",
        "top_2_prob",
        "top_3_label",
        "top_3_prob",
        "corrected_label",
        "review_decision",
        "review_note",
    ]

    existing_output_cols = [col for col in output_cols if col in combined_df.columns]
    combined_df = combined_df[existing_output_cols]

    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== Issue 错误样本复审文件生成完成 ==========")
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")
    print(f"原始测试集行数：{len(df)}")
    print(f"错误样本总数：{len(error_df)}")
    print(f"本次导出复审样本数：{len(combined_df)}")

    print("\n优先复审类别：")
    for label in PRIORITY_TRUE_LABELS:
        count = len(priority_df[priority_df["true_label"] == label])
        print(f"{label}: {count}")

    print("\n导出文件中的 true_label 分布：")
    print(combined_df["true_label"].value_counts())

    print("\n可用 corrected_label：")
    for label in VALID_ISSUE_LABELS:
        print(f"- {label}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/transformer_issue_test_predictions.csv",
        help="模型测试集预测明细文件路径",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/issue_error_review_candidates.csv",
        help="错误样本复审文件输出路径",
    )

    parser.add_argument(
        "--max_rows",
        type=int,
        default=150,
        help="最多导出的复审样本数量",
    )

    args = parser.parse_args()

    create_error_review_candidates(
        input_path=args.input_path,
        output_path=args.output_path,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()