import json
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENTIMENT_MODEL_DIR = PROJECT_ROOT / "models" / "transformer_sentiment"
ISSUE_MODEL_DIR = PROJECT_ROOT / "models" / "transformer_issue"
CLUSTER_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "clustering" / "cluster_summary_labeled.csv"


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


class FeedbackRequest(BaseModel):
    text: str = Field(..., description="Customer review text")
    top_k: int = Field(default=3, ge=1, le=10)
    min_candidate_confidence: float = Field(default=0.10, ge=0.0, le=1.0)
    dominant_threshold: float = Field(default=0.80, ge=0.0, le=1.0)


class CandidateIssue(BaseModel):
    rank: int
    label: str
    confidence: float


class FeedbackResponse(BaseModel):
    review_text: str
    sentiment_label: str
    sentiment_confidence: float
    primary_issue_label: str
    primary_issue_confidence: float
    issue_mode: str
    candidate_issues: List[CandidateIssue]
    ranked_issue_predictions: List[CandidateIssue]
    device: str


class ClusterItem(BaseModel):
    cluster_id: int
    cluster_theme: str
    cluster_size: int
    major_issue: str
    top_keywords: str
    representative_reviews: List[str]


class ClusterResponse(BaseModel):
    total_clusters: int
    clusters: List[ClusterItem]


def load_label_mapping(model_dir: Path, default_id2label: Dict[int, str]) -> Dict[int, str]:
    mapping_path = model_dir / "label_mapping.json"

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        id2label = mapping.get("id2label", default_id2label)
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        id2label = default_id2label

    return id2label


def load_model_and_tokenizer(model_dir: Path, device: torch.device):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    model.to(device)
    model.eval()

    return tokenizer, model


def predict_sentiment(
    text: str,
    tokenizer,
    model,
    id2label: Dict[int, str],
    device: torch.device,
    max_length: int = 128,
) -> Dict[str, Any]:
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

    return {
        "sentiment_label": predicted_label,
        "sentiment_confidence": confidence,
    }


def predict_issue(
    text: str,
    tokenizer,
    model,
    id2label: Dict[int, str],
    device: torch.device,
    max_length: int = 256,
    top_k: int = 3,
    min_candidate_confidence: float = 0.10,
    dominant_threshold: float = 0.80,
) -> Dict[str, Any]:
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
    primary_issue_confidence = ranked_predictions[0]["confidence"]

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

    if primary_issue_confidence >= dominant_threshold and len(candidate_issues) == 1:
        issue_mode = "single-dominant"

    return {
        "primary_issue_label": primary_issue_label,
        "primary_issue_confidence": primary_issue_confidence,
        "issue_mode": issue_mode,
        "candidate_issues": candidate_issues,
        "ranked_issue_predictions": ranked_predictions,
    }


app = FastAPI(
    title="Mobile Customer Feedback Analysis API",
    description="Transformer-based sentiment analysis, issue triage, and complaint cluster discovery API.",
    version="2.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentiment_id2label = load_label_mapping(
    SENTIMENT_MODEL_DIR,
    DEFAULT_SENTIMENT_ID2LABEL,
)

issue_id2label = load_label_mapping(
    ISSUE_MODEL_DIR,
    DEFAULT_ISSUE_ID2LABEL,
)

sentiment_tokenizer, sentiment_model = load_model_and_tokenizer(
    SENTIMENT_MODEL_DIR,
    device,
)

issue_tokenizer, issue_model = load_model_and_tokenizer(
    ISSUE_MODEL_DIR,
    device,
)


@app.get("/")
def root():
    return {
        "message": "Mobile Customer Feedback Analysis API is running.",
        "device": str(device),
        "sentiment_model": str(SENTIMENT_MODEL_DIR),
        "issue_model": str(ISSUE_MODEL_DIR),
        "cluster_summary": str(CLUSTER_SUMMARY_PATH),
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "sentiment_model_exists": SENTIMENT_MODEL_DIR.exists(),
        "issue_model_exists": ISSUE_MODEL_DIR.exists(),
        "cluster_summary_exists": CLUSTER_SUMMARY_PATH.exists(),
    }


@app.post("/predict", response_model=FeedbackResponse)
def predict_feedback(request: FeedbackRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    sentiment_result = predict_sentiment(
        text=text,
        tokenizer=sentiment_tokenizer,
        model=sentiment_model,
        id2label=sentiment_id2label,
        device=device,
        max_length=128,
    )

    issue_result = predict_issue(
        text=text,
        tokenizer=issue_tokenizer,
        model=issue_model,
        id2label=issue_id2label,
        device=device,
        max_length=256,
        top_k=request.top_k,
        min_candidate_confidence=request.min_candidate_confidence,
        dominant_threshold=request.dominant_threshold,
    )

    return {
        "review_text": text,
        "device": str(device),
        **sentiment_result,
        **issue_result,
    }


@app.get("/clusters", response_model=ClusterResponse)
def get_clusters(limit: int = 20):
    if not CLUSTER_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Cluster summary file not found: {CLUSTER_SUMMARY_PATH}",
        )

    try:
        df = pd.read_csv(CLUSTER_SUMMARY_PATH)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read cluster summary file: {str(e)}",
        )

    required_cols = [
        "cluster_id",
        "cluster_theme",
        "cluster_size",
        "major_issue",
        "top_keywords",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Cluster summary missing required column: {col}",
            )

    df = df.sort_values(by="cluster_size", ascending=False).head(limit)

    clusters = []

    for _, row in df.iterrows():
        representative_reviews = []

        for i in range(1, 4):
            col = f"representative_review_{i}"
            if col in df.columns:
                value = row.get(col, "")
                if isinstance(value, str) and value.strip():
                    representative_reviews.append(value.strip())

        clusters.append(
            {
                "cluster_id": int(row["cluster_id"]),
                "cluster_theme": str(row["cluster_theme"]),
                "cluster_size": int(row["cluster_size"]),
                "major_issue": str(row["major_issue"]),
                "top_keywords": str(row["top_keywords"]),
                "representative_reviews": representative_reviews,
            }
        )

    return {
        "total_clusters": len(clusters),
        "clusters": clusters,
    }