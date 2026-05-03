import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_ID2LABEL = {
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


def load_label_mapping(model_dir: str) -> dict[int, str]:
    mapping_path = Path(model_dir) / "label_mapping.json"

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        id2label = mapping.get("id2label", DEFAULT_ID2LABEL)
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        id2label = DEFAULT_ID2LABEL

    return id2label


def predict_issue(
    text: str,
    model_dir: str,
    max_length: int = 256,
    top_k: int = 3,
    min_candidate_confidence: float = 0.10,
    dominant_threshold: float = 0.80,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    id2label = load_label_mapping(model_dir)

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
        else:
            if item["confidence"] >= min_candidate_confidence:
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
        "primary_confidence": primary_confidence,
        "issue_mode": issue_mode,
        "candidate_issues": candidate_issues,
        "ranked_predictions": ranked_predictions,
        "all_probabilities": all_probabilities,
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input review text.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/transformer_issue",
        help="Path to the fine-tuned issue classifier.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum token length.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Maximum number of top issue labels to consider.",
    )

    parser.add_argument(
        "--min_candidate_confidence",
        type=float,
        default=0.10,
        help="Only show non-primary candidate issues above this confidence.",
    )

    parser.add_argument(
        "--dominant_threshold",
        type=float,
        default=0.80,
        help="If primary confidence is above this value, it is treated as a dominant issue.",
    )

    parser.add_argument(
        "--show_all",
        action="store_true",
        help="Show probabilities for all issue labels.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(
            f"找不到模型目录：{args.model_dir}\n"
            "请先运行 src/models/train_transformer_issue.py 训练 issue classifier。"
        )

    result = predict_issue(
        text=args.text,
        model_dir=args.model_dir,
        max_length=args.max_length,
        top_k=args.top_k,
        min_candidate_confidence=args.min_candidate_confidence,
        dominant_threshold=args.dominant_threshold,
    )

    print("\nInput review:")
    print(args.text)

    print("\nPrimary issue prediction:")
    print(f"primary_issue_label: {result['primary_issue_label']}")
    print(f"primary_confidence: {result['primary_confidence']:.4f}")
    print(f"issue_mode: {result['issue_mode']}")
    print(f"device: {result['device']}")

    print("\nCandidate issue predictions:")
    for item in result["candidate_issues"]:
        print(
            f"{item['rank']}. {item['label']}: "
            f"{item['confidence']:.4f}"
        )

    print(f"\nRaw Top-{args.top_k} predictions:")
    for item in result["ranked_predictions"]:
        print(
            f"{item['rank']}. {item['label']}: "
            f"{item['confidence']:.4f}"
        )

    if args.show_all:
        print("\nAll issue probabilities:")
        sorted_probs = sorted(
            result["all_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for label, prob in sorted_probs:
            print(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    main()