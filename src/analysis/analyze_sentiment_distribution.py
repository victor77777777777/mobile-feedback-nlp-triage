import pandas as pd
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "processed" / "clean_reviews.csv"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_summary_path = output_dir / "sentiment_distribution_summary.csv"

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    print("\nBasic dataset information:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nRating distribution:")
    rating_dist = df["rating"].value_counts(dropna=False).sort_index()
    print(rating_dist)

    print("\nSentiment label distribution:")
    sentiment_dist = df["sentiment_label"].value_counts(dropna=False)
    print(sentiment_dist)

    print("\nSentiment label proportion:")
    sentiment_prop = df["sentiment_label"].value_counts(normalize=True, dropna=False)
    print(sentiment_prop)

    summary = (
        df["sentiment_label"]
        .value_counts(dropna=False)
        .rename_axis("sentiment_label")
        .reset_index(name="count")
    )
    summary["proportion"] = summary["count"] / summary["count"].sum()

    summary.to_csv(output_summary_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved sentiment distribution summary to: {output_summary_path}")

    print("\nAverage rating by sentiment label:")
    avg_rating = df.groupby("sentiment_label")["rating"].mean()
    print(avg_rating)

    print("\nSample reviews by sentiment label:")
    for label in df["sentiment_label"].dropna().unique():
        print(f"\n===== {label} =====")
        samples = df[df["sentiment_label"] == label]["review_text"].dropna().head(3)

        for i, text in enumerate(samples, start=1):
            print(f"{i}. {text[:300]}...")


if __name__ == "__main__":
    main()