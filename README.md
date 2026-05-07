# Mobile Customer Feedback Analysis System

Author: ZHANG ZEYU / 张泽予

## 1. Project Overview

This project implements an intelligent customer feedback analysis system for mobile phone reviews. The system is designed to automatically analyze customer reviews, identify sentiment, classify complaint issues, discover recurring complaint themes, and provide priority-ranked product insights for product managers.

The final system includes both machine learning pipelines and an interactive dashboard:

- Sentiment classification
- Issue triage classification
- Top-K issue candidate prediction
- Complaint theme discovery using clustering
- Priority scoring for complaint clusters
- Product Manager insight report generation
- FastAPI backend service
- Browser-based frontend dashboard

The project focuses on the project implementation itself. Environment setup, repository creation, and other non-project configuration details are not included here.

---

## 2. Data Source and Dataset

The project uses a mobile phone customer review dataset. The cleaned dataset is stored as:

```text
data/processed/clean_reviews.csv
```

The cleaned data contains customer review records with fields such as:

```text
review_id
product_name
brand
price
rating
review_text
review_votes
sentiment_label
date
issue_label
```

The original `issue_label` field was empty in the cleaned dataset, so a separate issue annotation workflow was created. A sampled subset was generated for issue labeling, reviewed, corrected, and then used for training the transformer-based issue classifier.

Main annotated issue training file:

```text
data/annotated/issue_annotation_training_v2.csv
```

The issue label set includes:

```text
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
```

---

## 3. System Architecture

The final architecture contains five major layers:

```text
Raw / cleaned customer reviews
        ↓
Data cleaning and annotation utilities
        ↓
Transformer-based sentiment and issue classifiers
        ↓
Clustering, priority scoring, and report generation
        ↓
FastAPI backend + frontend dashboard
```

Main project folders:

```text
backend/        FastAPI backend service
frontend/       Browser-based dashboard
src/data/       Data cleaning and annotation scripts
src/models/     Model training scripts
src/inference/  Inference scripts
src/analysis/   Error analysis, clustering theme labeling, priority scoring
src/clustering/ Review clustering pipeline
src/reports/    Product Manager report generation
models/         Trained transformer models
outputs/        Reports, metrics, clustering results, and priority outputs
```

---

## 4. Implemented Modules

### 4.1 Data Cleaning

The project first cleaned raw mobile review data and generated a structured review dataset.

Main script:

```text
src/data/clean_data.py
```

Main output:

```text
data/processed/clean_reviews.csv
```

The cleaning process prepared review text, sentiment labels, product metadata, and review IDs for downstream modeling.

---

### 4.2 Baseline Sentiment Model

A TF-IDF + Logistic Regression baseline was implemented to provide a traditional machine learning comparison for sentiment classification.

Main outputs include:

```text
outputs/baseline_sentiment_report.txt
outputs/baseline_sentiment_confusion_matrix.csv
```

Baseline reference result:

```text
Accuracy = 0.7890
Macro-F1 = 0.6660
```

This baseline provided a useful benchmark before introducing transformer-based models.

---

### 4.3 Transformer Sentiment Classifier

A DistilBERT-based sentiment classifier was implemented for three-class sentiment classification:

```text
negative
neutral
positive
```

Main training script:

```text
src/models/train_transformer_sentiment.py
```

Main inference script:

```text
src/inference/predict_transformer_sentiment.py
```

Saved model directory:

```text
models/transformer_sentiment
```

The transformer sentiment model supports GPU acceleration and can be called independently or through the unified inference pipeline.

Example output:

```text
Input review:
The battery dies very quickly and the phone freezes all the time.

Transformer sentiment prediction:
sentiment_label: negative
confidence: 0.7038
device: cuda
```

---

### 4.4 Issue Annotation Workflow

Since the original dataset did not contain issue labels, an issue annotation workflow was created.

Main scripts:

```text
src/data/create_annotation_sample.py
src/data/bootstrap_issue_labels.py
src/analysis/create_issue_error_review_candidates.py
```

The workflow included:

1. Sampling reviews for annotation
2. Generating weak/bootstrap issue labels
3. Manually reviewing confusing or wrong labels
4. Merging corrected labels into a revised training set
5. Re-training the issue classifier

The final reviewed issue training file is:

```text
data/annotated/issue_annotation_training_v2.csv
```

During manual review, several label boundary rules were clarified. For example:

- SIM lock, carrier lock, LTE/4G, Verizon, Sprint, T-Mobile, and activation issues are treated as `network`.
- Charging, battery drain, charger, USB charging port, and overheating during charging are treated as `battery`.
- Refurbished, used, scratched, damaged, broken-on-arrival, and poor build quality issues are treated as `condition`.
- UI behavior, software update, app compatibility, root, localization, and manual/instruction issues are treated as `software`.
- Broad positive reviews without a clear issue are treated as `other`.

---

### 4.5 Transformer Issue Classifier

A DistilBERT-based issue classifier was implemented using the reviewed issue training data.

Main training script:

```text
src/models/train_transformer_issue.py
```

Main inference script:

```text
src/inference/predict_transformer_issue.py
```

Saved model directory:

```text
models/transformer_issue
```

The first training result with shorter text truncation had limited performance. After increasing `max_length` from 160 to 256, the issue classifier improved significantly.

Final issue classifier result:

```text
Accuracy = 0.7275
Macro-F1 = 0.6840
Weighted-F1 = 0.7118
```

The improvement showed that longer review context is important because many customer reviews contain multiple complaints, and the main issue may appear later in the review.

---

### 4.6 Top-K Issue Candidate Prediction

A Top-K issue prediction mechanism was added to avoid only returning a single issue label.

Main script:

```text
src/inference/predict_transformer_issue.py
```

The logic includes:

```text
primary_issue_label
primary_confidence
issue_mode
candidate_issues
ranked_issue_predictions
```

Default values:

```text
top_k = 3
min_candidate_confidence = 0.10
dominant_threshold = 0.80
```

Important note:

The current issue classifier is still a single-label softmax classifier. Therefore, Top-K outputs should be interpreted as candidate ranking scores rather than true independent multi-label probabilities.

---

### 4.7 Unified Feedback Inference

A unified inference script was created to combine sentiment classification and issue triage.

Main script:

```text
src/inference/predict_feedback_v2.py
```

The unified output includes:

```text
sentiment_label
sentiment_confidence
primary_issue_label
primary_issue_confidence
issue_mode
candidate_issues
device
```

Example:

```text
Input review:
The phone is locked and cannot be used with my SIM card.

sentiment_label: negative
sentiment_confidence: 0.9496
primary_issue_label: network
primary_issue_confidence: 0.8267
issue_mode: single-dominant
device: cuda
```

---

## 5. Clustering and Topic Discovery

A clustering module was implemented to discover recurring complaint themes from negative reviews.

Main script:

```text
src/clustering/cluster_reviews.py
```

Method:

```text
sentence-transformers/all-MiniLM-L6-v2
+ MiniBatchKMeans
+ TF-IDF keyword extraction
+ representative review selection
```

The clustering pipeline:

1. Reads cleaned mobile reviews
2. Filters negative reviews
3. Predicts issue labels using the transformer issue classifier
4. Generates sentence embeddings
5. Applies MiniBatchKMeans clustering
6. Extracts cluster keywords using TF-IDF
7. Selects representative reviews closest to cluster centers
8. Saves cluster-level and review-level outputs

Main outputs:

```text
outputs/clustering/review_clusters.csv
outputs/clustering/cluster_summary.csv
outputs/clustering/cluster_summary_labeled.csv
outputs/clustering/clustering_report.md
```

Final clustering run:

```text
Clustered reviews: 20,000 negative reviews
Number of clusters: 20
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Clustering method: MiniBatchKMeans
```

Discovered themes included:

```text
Battery and charging complaints
Carrier, SIM, and network compatibility
Performance, lag, and app stability
Refurbished, used, or damaged condition
Screen and display defects
Audio, speaker, and call sound issues
Accessory or case compatibility issues
Smartwatch connectivity and quality issues
General quality dissatisfaction
```

Representative reviews were added to make clusters interpretable and useful for product managers.

---

## 6. Cluster Theme Labeling

After clustering, cluster IDs were automatically mapped to human-readable themes.

Main script:

```text
src/analysis/label_cluster_themes.py
```

Input:

```text
outputs/clustering/cluster_summary.csv
```

Output:

```text
outputs/clustering/cluster_summary_labeled.csv
```

The labeling step used a rule-based mapping based on:

```text
top_keywords
major_issue
representative reviews
```

Some cluster themes were manually refined to better match business interpretation.

---

## 7. Clustering Report Generation

A clustering report was generated to summarize discovered complaint topics.

Main script:

```text
src/analysis/generate_clustering_report.py
```

Output:

```text
outputs/clustering/clustering_report.md
```

The report includes:

```text
Overview
Cluster theme distribution
Top clusters by size
Detailed cluster summaries
Representative reviews
Interpretation and limitations
```

The report explains that customer reviews often contain multiple issues in one paragraph, so clusters should be interpreted as practical complaint themes rather than strictly separated semantic groups.

---

## 8. Priority Scoring

A priority scoring module was added to rank complaint themes by business importance.

Main script:

```text
src/analysis/calculate_priority_scores.py
```

Input:

```text
outputs/clustering/cluster_summary_labeled.csv
```

Output:

```text
outputs/priority/priority_scores.csv
```

Priority score combines:

```text
cluster_size_score
issue_concentration_score
confidence_score
negative_ratio_score
growth_proxy_score
```

Formula:

```text
priority_score =
0.40 * cluster_size_score
+ 0.25 * issue_concentration_score
+ 0.20 * confidence_score
+ 0.10 * negative_ratio_score
+ 0.05 * growth_proxy_score
```

Because the current public dataset does not contain reliable timestamps, real short-term trend growth could not be calculated. A neutral growth proxy was used, and time-based trend scoring is left as future work.

Top priority themes included:

```text
Battery and charging complaints
Carrier, SIM, and network compatibility
General quality dissatisfaction
Performance, lag, and app stability
```

---

## 9. Product Manager Insight Report

A Product Manager insight report generation module was implemented.

Main script:

```text
src/reports/generate_pm_insight_report.py
```

Input:

```text
outputs/priority/priority_scores.csv
```

Output:

```text
outputs/reports/pm_insight_report.md
```

The report contains:

```text
Executive Summary
Top Priority Complaint Themes
Detailed Priority Analysis
Representative Customer Evidence
Recommended Product Actions
Product Implications
Limitations and Future Work
```

Example recommended actions include:

- Investigate battery, charging, charger, and USB-port related failure patterns.
- Review carrier, SIM, LTE/4G, unlocked status, and regional compatibility information.
- Strengthen refurbished-device inspection and product listing transparency.
- Analyze slow response, app crashes, freezing, unexpected shutdowns, and early device failure.
- Consider adding dedicated audio and accessory issue categories in the next taxonomy version.

---

## 10. FastAPI Backend

A backend API service was created with FastAPI.

Main file:

```text
backend/app.py
```

Main endpoints:

```text
GET  /
GET  /health
POST /predict
GET  /clusters
GET  /priority
GET  /pm-report
```

### POST /predict

Input:

```json
{
  "text": "The phone freezes all the time and the battery dies quickly.",
  "top_k": 3,
  "min_candidate_confidence": 0.10,
  "dominant_threshold": 0.80
}
```

Output:

```json
{
  "sentiment_label": "negative",
  "sentiment_confidence": 0.7091,
  "primary_issue_label": "battery",
  "primary_issue_confidence": 0.7837,
  "issue_mode": "single",
  "candidate_issues": [...]
}
```

### GET /clusters

Returns complaint cluster summaries from:

```text
outputs/clustering/cluster_summary_labeled.csv
```

### GET /priority

Returns priority-ranked complaint themes from:

```text
outputs/priority/priority_scores.csv
```

### GET /pm-report

Returns the generated Product Manager insight report from:

```text
outputs/reports/pm_insight_report.md
```

---

## 11. Frontend Dashboard

A browser-based dashboard was created.

Main file:

```text
frontend/index.html
```

Dashboard modules:

```text
Single Review Analysis
Complaint Theme Discovery
Top Priority Issues
Product Manager Insight Report
```

The dashboard can:

1. Analyze one customer review
2. Display sentiment and issue prediction
3. Show Top-K issue candidates
4. Load complaint clusters
5. Load priority-ranked issues
6. Display the generated PM insight report

Default frontend values:

```text
Top-K = 3
Cluster Limit = 5
Priority Limit = 5
```

---

## 12. How to Run the Project Demo

Start the FastAPI backend:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Open API documentation:

```text
http://127.0.0.1:8000/docs
```

Open the frontend dashboard:

```text
frontend/index.html
```

Or open it directly in the browser from:

```text
E:\nlp project\frontend\index.html
```

---

## 13. Main Results

### Sentiment Classification

The project implemented both baseline and transformer sentiment classifiers.

Baseline reference:

```text
TF-IDF + Logistic Regression
Accuracy = 0.7890
Macro-F1 = 0.6660
```

Transformer sentiment model supports GPU inference and integrates with the final dashboard.

### Issue Classification

Final DistilBERT issue classifier result after reviewed labels and longer max sequence length:

```text
Accuracy = 0.7275
Macro-F1 = 0.6840
Weighted-F1 = 0.7118
```

The max sequence length improvement from 160 to 256 significantly improved performance, especially for long reviews.

### Clustering

```text
20,000 negative reviews
20 clusters
sentence-transformers/all-MiniLM-L6-v2 embeddings
MiniBatchKMeans clustering
TF-IDF keyword extraction
Representative review selection
```

### Priority Scoring and PM Report

The system generates:

```text
outputs/priority/priority_scores.csv
outputs/reports/pm_insight_report.md
```

These outputs convert model and clustering results into product-level insights.

---

## 14. Limitations

Current limitations:

1. The issue classifier is a single-label classifier, so Top-K outputs are candidate rankings rather than true multi-label probabilities.
2. Some customer reviews mention multiple issues in one paragraph, which makes both classification and clustering more difficult.
3. The dataset does not provide reliable timestamps, so real trend growth cannot yet be calculated.
4. The current label schema does not include dedicated `audio` or `accessory` labels, so related complaints are mapped to broader categories.
5. Clustering uses unsupervised semantic grouping, so clusters should be interpreted as practical complaint themes, not perfectly separated categories.

---

## 15. Future Work

Possible next steps:

1. Convert issue classification from single-label softmax to true multi-label sigmoid classification.
2. Add dedicated labels such as `audio`, `accessory`, and `hardware`.
3. Add time-based trend detection when timestamped data is available.
4. Add batch upload support to the dashboard.
5. Improve Markdown rendering for PM reports in the frontend.
6. Add charts for issue distribution, priority score distribution, and cluster size distribution.
7. Deploy the backend and frontend as a complete web service.
8. Integrate the system with real company feedback channels or support tickets.

---

## 16. Final Summary

This project has evolved from a basic sentiment analysis pipeline into a complete mobile customer feedback intelligence system.

The final system can:

- classify customer sentiment,
- identify the primary product issue,
- provide Top-K issue candidates,
- discover recurring complaint themes,
- rank complaint clusters by priority,
- generate Product Manager insight reports,
- and expose all major functions through a FastAPI backend and frontend dashboard.

The implementation demonstrates how traditional machine learning, transformer models, sentence embedding clustering, rule-based business logic, and dashboard integration can be combined into a practical customer feedback triage system.
