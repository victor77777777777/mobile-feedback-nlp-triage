import argparse
from pathlib import Path

import pandas as pd


THEME_RULES = [
    {
        "theme": "Battery and charging complaints",
        "keywords": [
            "battery", "charge", "charging", "charger", "drain",
            "power", "turn", "dead", "dies", "hot", "overheat",
            "usb", "connector"
        ],
        "issues": ["battery"],
    },
    {
        "theme": "Carrier, SIM, and network compatibility",
        "keywords": [
            "unlocked", "locked", "verizon", "sprint", "att", "at&t",
            "tmobile", "t-mobile", "sim", "card", "lte", "4g", "3g",
            "network", "carrier", "wifi", "wi-fi", "data", "activate",
            "activation", "international", "venezuela", "brazil", "mobile"
        ],
        "issues": ["network"],
    },
    {
        "theme": "Screen and display defects",
        "keywords": [
            "screen", "display", "black", "touch", "touchscreen",
            "cracked", "crack", "glass", "lcd", "stripe", "pixel",
            "brightness", "responsive"
        ],
        "issues": ["screen"],
    },
    {
        "theme": "Performance, lag, and app stability",
        "keywords": [
            "slow", "lag", "laggy", "freeze", "freezes", "frozen",
            "crash", "crashes", "apps", "app", "stuck", "hang",
            "response", "responsive", "restart", "restarts"
        ],
        "issues": ["performance"],
    },
    {
        "theme": "Refurbished, used, or damaged condition",
        "keywords": [
            "refurbished", "used", "scratch", "scratches", "worn",
            "dent", "dented", "broken", "damage", "damaged", "new",
            "condition", "original", "back", "case", "dirty", "old"
        ],
        "issues": ["condition"],
    },
    {
        "theme": "Camera and photo quality issues",
        "keywords": [
            "camera", "photo", "photos", "picture", "pictures",
            "video", "focus", "blurry", "blur", "flash", "front",
            "quality"
        ],
        "issues": ["camera"],
    },
    {
        "theme": "Shipping, delivery, and packaging issues",
        "keywords": [
            "shipping", "delivery", "delivered", "arrived", "late",
            "package", "packaging", "box", "missing", "received",
            "send", "sent"
        ],
        "issues": ["shipping"],
    },
    {
        "theme": "Seller, return, refund, and warranty service",
        "keywords": [
            "seller", "service", "support", "return", "refund",
            "warranty", "repair", "replacement", "claim", "respond",
            "response", "company", "customer"
        ],
        "issues": ["service"],
    },
    {
        "theme": "Price and value dissatisfaction",
        "keywords": [
            "price", "expensive", "cheap", "value", "money",
            "worth", "cost", "deal", "paid", "pay"
        ],
        "issues": ["price"],
    },
    {
        "theme": "Software, update, UI, and setup issues",
        "keywords": [
            "software", "update", "android", "system", "os", "ui",
            "settings", "setting", "root", "rooted", "bloatware",
            "manual", "instructions", "sync", "program", "language",
            "chinese", "keyboard"
        ],
        "issues": ["software"],
    },
    {
        "theme": "Audio, speaker, and call sound issues",
        "keywords": [
            "speaker", "sound", "volume", "hear", "audio",
            "microphone", "mic", "call", "calls", "voice",
            "headphone", "headphones"
        ],
        "issues": [],
    },
    {
        "theme": "Accessory or case compatibility issues",
        "keywords": [
            "otterbox", "cover", "case", "fit", "protector",
            "accessory", "accessories", "cable", "usb", "adapter"
        ],
        "issues": [],
    },
]


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).lower()


def score_theme(row: pd.Series, rule: dict) -> int:
    text_parts = [
        normalize_text(row.get("top_keywords", "")),
        normalize_text(row.get("major_issue", "")),
        normalize_text(row.get("representative_review_1", "")),
        normalize_text(row.get("representative_review_2", "")),
        normalize_text(row.get("representative_review_3", "")),
    ]

    combined_text = " ".join(text_parts)

    score = 0

    for keyword in rule["keywords"]:
        if keyword.lower() in combined_text:
            score += 1

    major_issue = normalize_text(row.get("major_issue", ""))

    if major_issue in rule["issues"]:
        score += 3

    return score


def assign_cluster_theme(row: pd.Series) -> tuple[str, int]:
    best_theme = "General negative feedback / mixed complaints"
    best_score = 0

    for rule in THEME_RULES:
        score = score_theme(row, rule)

        if score > best_score:
            best_score = score
            best_theme = rule["theme"]

    return best_theme, best_score


def label_cluster_themes(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_cols = [
        "cluster_id",
        "cluster_size",
        "top_keywords",
        "major_issue",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入文件缺少必要列：{col}")

    themes = []
    theme_scores = []

    for _, row in df.iterrows():
        theme, score = assign_cluster_theme(row)
        themes.append(theme)
        theme_scores.append(score)

    df.insert(1, "cluster_theme", themes)
    df.insert(2, "theme_confidence_score", theme_scores)

    first_cols = [
        "cluster_id",
        "cluster_theme",
        "theme_confidence_score",
        "cluster_size",
        "top_keywords",
        "major_issue",
        "major_issue_count",
        "avg_issue_confidence",
        "negative_ratio",
    ]

    existing_first_cols = [col for col in first_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_first_cols]

    df = df[existing_first_cols + remaining_cols]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== Cluster theme labeling 完成 ==========")
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")
    print(f"Cluster 数量：{len(df)}")

    print("\nCluster theme 分布：")
    print(df["cluster_theme"].value_counts())

    print("\nTop clusters:")
    display_cols = [
        "cluster_id",
        "cluster_theme",
        "cluster_size",
        "top_keywords",
        "major_issue",
    ]
    display_cols = [col for col in display_cols if col in df.columns]

    print(df[display_cols].head(20))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/clustering/cluster_summary.csv",
        help="原始 cluster summary 文件路径",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/clustering/cluster_summary_labeled.csv",
        help="带主题名的 cluster summary 输出路径",
    )

    args = parser.parse_args()

    label_cluster_themes(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()