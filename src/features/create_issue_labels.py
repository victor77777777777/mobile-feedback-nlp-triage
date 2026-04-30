import pandas as pd
from pathlib import Path


ISSUE_KEYWORDS = {
    "battery": [
        "battery", "charge", "charging", "charged", "charger",
        "power", "drain", "drained", "dies", "dead", "last long"
    ],
    "screen": [
        "screen", "display", "touch", "touchscreen", "black screen",
        "crack", "cracked", "scratch", "scratched", "glass"
    ],
    "performance": [
        "slow", "lag", "laggy", "freeze", "freezes", "frozen",
        "crash", "crashes", "restart", "reboot", "overheat", "hot"
    ],
    "shipping": [
        "shipping", "delivery", "delivered", "arrived", "package",
        "packaging", "box", "late", "delay", "shipment"
    ],
    "condition": [
        "used", "refurbished", "condition", "damaged", "defective",
        "broken", "dirty", "worn", "dent", "dented"
    ],
    "camera": [
        "camera", "photo", "picture", "video", "lens", "flash"
    ],
    "service": [
        "seller", "refund", "return", "replacement", "warranty",
        "customer service", "support", "contacted"
    ],
    "network": [
        "sim", "carrier", "network", "signal", "wifi", "bluetooth",
        "activation", "activate", "unlocked", "locked", "sprint",
        "verizon", "at&t", "tmobile", "t-mobile"
    ],
    "price": [
        "price", "cost", "expensive", "cheap", "money", "worth",
        "value", "paid", "pay"
    ],
}


def assign_issue_label(text):
    if pd.isna(text):
        return "other"

    text = str(text).lower()

    matched_labels = []

    for issue, keywords in ISSUE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                matched_labels.append(issue)
                break

    if not matched_labels:
        return "other"

    return matched_labels[0]


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "processed" / "clean_reviews.csv"
    output_path = project_root / "data" / "processed" / "clean_reviews_with_issues.csv"

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    print("\nAssigning issue labels...")
    df["issue_label"] = df["review_text"].apply(assign_issue_label)

    print("\nIssue label distribution:")
    issue_dist = df["issue_label"].value_counts()
    print(issue_dist)

    print("\nIssue label proportion:")
    issue_prop = df["issue_label"].value_counts(normalize=True)
    print(issue_prop)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved labeled data to: {output_path}")

    print("\nSample reviews by issue label:")
    for label in df["issue_label"].value_counts().index:
        print(f"\n===== {label} =====")
        samples = df[df["issue_label"] == label]["review_text"].dropna().head(3)

        for i, text in enumerate(samples, start=1):
            print(f"{i}. {text[:300]}...")


if __name__ == "__main__":
    main()