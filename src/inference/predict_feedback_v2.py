import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_SENTIMENT_ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


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


def load_label_mapping(model_dir: str, default_id2label: dict[int, str]) -> dict[int, str]:
    mapping_path = Path(model_dir) / "label_mapping.json"

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        id2label = mapping.get("id2label", default_id2label)
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        id2label = default_id2label

    return id2label


def load_model_and_tokenizer(model_dir: str, device: torch.device):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"找不到模型目录：{model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    return tokenizer, model


def predict_sentiment(
    text: str,
    tokenizer,
    model,
    id2label: dict[int, str],
    device: torch.device,
    max_length: int = 128,
):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    predicted_label = id2label[predicted_id]
    confidence = float(probabilities[predicted_id].item())

    all_probabilities = {
        id2label[i]: float(probabilities[i].item())
        for i in range(len(probabilities))
    }

    return {
        "sentiment_label": predicted_label,
        "sentiment_confidence": confidence,
        "sentiment_probabilities": all_probabilities,
    }


def predict_issue(
    text: str,
    tokenizer,
    model,
    id2label: dict[int, str],
    device: torch.device,
    max_length: int = 256,
    top_k: int = 3,
    min_candidate_confidence: float = 0.10,
    dominant_threshold: float = 0.80,
):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)[0]

    top_k = min(top_k, len(probabilities))
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    ranked_predictions = []

    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        label_id = int(idx.item())
        label = id2label[label_id]
        confidence = float(prob.item())

        ranked_predictions.append(
            {
                "rank": rank,
                "label": label,
                "confidence": confidence,
            }
        )

    primary_issue_label = ranked_predictions[0]["label"]
    primary_confidence = ranked_predictions[0]["confidence"]

    candidate_issues = []

    for item in ranked_predictions:
        if item["rank"] == 1:
            candidate_issues.append(item)
        elif item["confidence"] >= min_candidate_confidence:
            candidate_issues.append(item)

    if len(candidate_issues) == 1:
        issue_mode = "single"
    else:
        issue_mode = "multi-candidate"

    if primary_confidence >= dominant_threshold and len(candidate_issues) == 1:
        issue_mode = "single-dominant"

    all_probabilities = {
        id2label[i]: float(probabilities[i].item())
        for i in range(len(probabilities))
    }

    return {
        "primary_issue_label": primary_issue_label,
        "primary_issue_confidence": primary_confidence,
        "issue_mode": issue_mode,
        "candidate_issues": candidate_issues,
        "ranked_issue_predictions": ranked_predictions,
        "issue_probabilities": all_probabilities,
    }


def predict_feedback(
    text: str,
    sentiment_model_dir: str,
    issue_model_dir: str,
    sentiment_max_length: int = 128,
    issue_max_length: int = 256,
    top_k: int = 3,
    min_candidate_confidence: float = 0.10,
    dominant_threshold: float = 0.80,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sentiment_id2label = load_label_mapping(
        model_dir=sentiment_model_dir,
        default_id2label=DEFAULT_SENTIMENT_ID2LABEL,
    )

    issue_id2label = load_label_mapping(
        model_dir=issue_model_dir,
        default_id2label=DEFAULT_ISSUE_ID2LABEL,
    )

    sentiment_tokenizer, sentiment_model = load_model_and_tokenizer(
        model_dir=sentiment_model_dir,
        device=device,
    )

    issue_tokenizer, issue_model = load_model_and_tokenizer(
        model_dir=issue_model_dir,
        device=device,
    )

    sentiment_result = predict_sentiment(
        text=text,
        tokenizer=sentiment_tokenizer,
        model=sentiment_model,
        id2label=sentiment_id2label,
        device=device,
        max_length=sentiment_max_length,
    )

    issue_result = predict_issue(
        text=text,
        tokenizer=issue_tokenizer,
        model=issue_model,
        id2label=issue_id2label,
        device=device,
        max_length=issue_max_length,
        top_k=top_k,
        min_candidate_confidence=min_candidate_confidence,
        dominant_threshold=dominant_threshold,
    )

    return {
        "review_text": text,
        "device": str(device),
        **sentiment_result,
        **issue_result,
    }


def print_prediction_result(result: dict, show_all: bool = False):
    print("\nInput review:")
    print(result["review_text"])

    print("\nFeedback prediction result:")
    print(f"sentiment_label: {result['sentiment_label']}")
    print(f"sentiment_confidence: {result['sentiment_confidence']:.4f}")

    print(f"primary_issue_label: {result['primary_issue_label']}")
    print(f"primary_issue_confidence: {result['primary_issue_confidence']:.4f}")
    print(f"issue_mode: {result['issue_mode']}")
    print(f"device: {result['device']}")

    print("\nCandidate issue predictions:")
    for item in result["candidate_issues"]:
        print(
            f"{item['rank']}. {item['label']}: "
            f"{item['confidence']:.4f}"
        )

    print("\nRaw issue Top-K predictions:")
    for item in result["ranked_issue_predictions"]:
        print(
            f"{item['rank']}. {item['label']}: "
            f"{item['confidence']:.4f}"
        )

    if show_all:
        print("\nSentiment probabilities:")
        sorted_sentiments = sorted(
            result["sentiment_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for label, prob in sorted_sentiments:
            print(f"{label}: {prob:.4f}")

        print("\nIssue probabilities:")
        sorted_issues = sorted(
            result["issue_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for label, prob in sorted_issues:
            print(f"{label}: {prob:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input customer review text.",
    )

    parser.add_argument(
        "--sentiment_model_dir",
        type=str,
        default="models/transformer_sentiment",
        help="Path to fine-tuned sentiment classifier.",
    )

    parser.add_argument(
        "--issue_model_dir",
        type=str,
        default="models/transformer_issue",
        help="Path to fine-tuned issue classifier.",
    )

    parser.add_argument(
        "--sentiment_max_length",
        type=int,
        default=128,
        help="Maximum token length for sentiment model.",
    )

    parser.add_argument(
        "--issue_max_length",
        type=int,
        default=256,
        help="Maximum token length for issue model.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Maximum number of issue candidates to consider.",
    )

    parser.add_argument(
        "--min_candidate_confidence",
        type=float,
        default=0.10,
        help="Only show non-primary issue candidates above this confidence.",
    )

    parser.add_argument(
        "--dominant_threshold",
        type=float,
        default=0.80,
        help="If primary issue confidence is above this value, treat it as dominant.",
    )

    parser.add_argument(
        "--show_all",
        action="store_true",
        help="Show all sentiment and issue probabilities.",
    )

    args = parser.parse_args()

    result = predict_feedback(
        text=args.text,
        sentiment_model_dir=args.sentiment_model_dir,
        issue_model_dir=args.issue_model_dir,
        sentiment_max_length=args.sentiment_max_length,
        issue_max_length=args.issue_max_length,
        top_k=args.top_k,
        min_candidate_confidence=args.min_candidate_confidence,
        dominant_threshold=args.dominant_threshold,
    )

    print_prediction_result(
        result=result,
        show_all=args.show_all,
    )


if __name__ == "__main__":
    main()