import argparse
from pathlib import Path

import joblib
import pandas as pd


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


def assign_issue_label(text: str) -> str:
    if pd.isna(text):
        return "other"

    text = str(text).lower()

    for issue, keywords in ISSUE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return issue

    return "other"


def predict_sentiment(text: str, model, vectorizer) -> str:
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return prediction


def main():
    parser = argparse.ArgumentParser(
        description="Predict sentiment and issue label for a mobile phone review."
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input review text.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    model_path = project_root / "models" / "baseline_sentiment_model.joblib"
    vectorizer_path = project_root / "models" / "tfidf_vectorizer.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please run src/models/train_baseline_sentiment.py first."
        )

    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found: {vectorizer_path}. "
            "Please run src/models/train_baseline_sentiment.py first."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    sentiment_label = predict_sentiment(args.text, model, vectorizer)
    issue_label = assign_issue_label(args.text)

    print("\nInput review:")
    print(args.text)

    print("\nPrediction result:")
    print(f"sentiment_label: {sentiment_label}")
    print(f"issue_label: {issue_label}")


if __name__ == "__main__":
    main()