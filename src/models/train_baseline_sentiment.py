import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "processed" / "clean_reviews.csv"
    model_dir = project_root / "models"
    output_dir = project_root / "outputs"

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "baseline_sentiment_model.joblib"
    vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
    report_path = output_dir / "baseline_sentiment_report.txt"
    confusion_matrix_path = output_dir / "baseline_sentiment_confusion_matrix.csv"

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    df = df.dropna(subset=["review_text", "sentiment_label"])

    X = df["review_text"].astype(str)
    y = df["sentiment_label"].astype(str)

    print("\nDataset size:")
    print(f"Total samples: {len(df)}")

    print("\nLabel distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/Test split:")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        stop_words="english",
    )

    print("\nFitting TF-IDF vectorizer...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["negative", "neutral", "positive"],
    )

    print("\nAccuracy:")
    print(acc)

    print("\nClassification report:")
    print(report)

    cm_df = pd.DataFrame(
        cm,
        index=["true_negative", "true_neutral", "true_positive"],
        columns=["pred_negative", "pred_neutral", "pred_positive"],
    )

    print("\nConfusion matrix:")
    print(cm_df)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Baseline Sentiment Classification Report\n")
        f.write("=======================================\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    cm_df.to_csv(confusion_matrix_path, encoding="utf-8-sig")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")


if __name__ == "__main__":
    main()