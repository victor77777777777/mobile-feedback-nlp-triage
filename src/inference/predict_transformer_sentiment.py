import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_label_mapping(model_dir: str):
    """
    Load label mapping saved during training.
    If label_mapping.json does not exist, use the default mapping.
    """
    mapping_path = Path(model_dir) / "label_mapping.json"

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        id2label = mapping.get("id2label", {})

        # JSON keys are strings, convert them back to integers
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        id2label = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }

    return id2label


def predict_sentiment(text: str, model_dir: str, max_length: int = 128):
    """
    Predict sentiment for one review using the fine-tuned transformer model.
    """
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

    predicted_id = int(torch.argmax(probabilities).item())
    predicted_label = id2label[predicted_id]
    confidence = float(probabilities[predicted_id].item())

    prob_dict = {
        id2label[i]: float(probabilities[i].item())
        for i in range(len(probabilities))
    }

    return {
        "sentiment_label": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input review text."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/transformer_sentiment",
        help="Path to the fine-tuned transformer sentiment model."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token length."
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {args.model_dir}\n"
            "Please train the model first using src/models/train_transformer_sentiment.py"
        )

    result = predict_sentiment(
        text=args.text,
        model_dir=args.model_dir,
        max_length=args.max_length,
    )

    print("\nInput review:")
    print(args.text)

    print("\nTransformer sentiment prediction:")
    print(f"sentiment_label: {result['sentiment_label']}")
    print(f"confidence: {result['confidence']:.4f}")
    print(f"device: {result['device']}")

    print("\nClass probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    main()