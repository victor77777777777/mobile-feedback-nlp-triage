import pandas as pd
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "processed" / "clean_reviews_with_issues.csv"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    issue_summary_path = output_dir / "issue_distribution_summary.csv"
    issue_sentiment_path = output_dir / "issue_sentiment_crosstab.csv"
    negative_issue_path = output_dir / "negative_issue_distribution.csv"

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    print("\nBasic dataset information:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df[["sentiment_label", "issue_label", "review_text"]].isna().sum())

    print("\nIssue label distribution:")
    issue_counts = df["issue_label"].value_counts(dropna=False)
    issue_props = df["issue_label"].value_counts(normalize=True, dropna=False)

    issue_summary = pd.DataFrame({
        "issue_label": issue_counts.index,
        "count": issue_counts.values,
        "proportion": issue_props.values,
    })

    print(issue_summary)
    issue_summary.to_csv(issue_summary_path, index=False, encoding="utf-8-sig")

    print("\nSentiment distribution within each issue label:")
    issue_sentiment = pd.crosstab(
        df["issue_label"],
        df["sentiment_label"],
        normalize="index",
    )

    issue_sentiment_counts = pd.crosstab(
        df["issue_label"],
        df["sentiment_label"],
    )

    print(issue_sentiment)
    issue_sentiment.to_csv(issue_sentiment_path, encoding="utf-8-sig")

    print("\nSentiment count table by issue label:")
    print(issue_sentiment_counts)

    print("\nTop issue labels among negative reviews:")
    negative_df = df[df["sentiment_label"] == "negative"]

    negative_issue_counts = negative_df["issue_label"].value_counts()
    negative_issue_props = negative_df["issue_label"].value_counts(normalize=True)

    negative_issue_summary = pd.DataFrame({
        "issue_label": negative_issue_counts.index,
        "negative_count": negative_issue_counts.values,
        "negative_proportion": negative_issue_props.values,
    })

    print(negative_issue_summary)
    negative_issue_summary.to_csv(
        negative_issue_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSample negative reviews by issue label:")
    for label in negative_issue_counts.index:
        print(f"\n===== {label} =====")
        samples = (
            negative_df[negative_df["issue_label"] == label]["review_text"]
            .dropna()
            .head(3)
        )

        for i, text in enumerate(samples, start=1):
            print(f"{i}. {text[:300]}...")

    print(f"\nSaved issue summary to: {issue_summary_path}")
    print(f"Saved issue-sentiment crosstab to: {issue_sentiment_path}")
    print(f"Saved negative issue distribution to: {negative_issue_path}")


if __name__ == "__main__":
    main()