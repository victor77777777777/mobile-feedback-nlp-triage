import re
from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/reviews.csv")
OUT_PATH = Path("data/processed/clean_reviews.csv")


def clean_text(text: str) -> str:
    """
    Clean raw review text.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"<.*?>", " ", text)           # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove URLs
    text = re.sub(r"\s+", " ", text)             # normalize spaces
    return text.strip()


def rating_to_sentiment(rating):
    """
    Convert 1-5 star rating to sentiment label.
    1-2 stars -> negative
    3 stars   -> neutral
    4-5 stars -> positive
    """
    try:
        rating = float(rating)
    except Exception:
        return None

    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the Amazon mobile phone dataset columns into our project format.

    Expected original columns are usually:
    Product Name, Brand Name, Price, Rating, Reviews, Review Votes
    """

    print("Original columns:")
    print(df.columns.tolist())

    column_mapping = {
        "Product Name": "product_name",
        "Brand Name": "brand",
        "Price": "price",
        "Rating": "rating",
        "Reviews": "review_text",
        "Review Votes": "review_votes",
    }

    df = df.rename(columns=column_mapping)

    required_columns = ["review_text", "rating"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column: {col}. "
                f"Current columns are: {df.columns.tolist()}"
            )

    return df


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {RAW_PATH}. Please put your CSV file at data/raw/reviews.csv"
        )

    df = pd.read_csv(RAW_PATH)

    df = standardize_columns(df)

    # Keep useful columns if they exist
    keep_columns = [
        "product_name",
        "brand",
        "price",
        "rating",
        "review_text",
        "review_votes",
    ]

    existing_columns = [col for col in keep_columns if col in df.columns]
    df = df[existing_columns].copy()

    # Clean review text
    df["review_text"] = df["review_text"].apply(clean_text)

    # Remove empty or very short reviews
    df = df[df["review_text"].str.len() > 5].copy()

    # Remove duplicated reviews
    df = df.drop_duplicates(subset=["review_text"]).copy()

    # Add review id
    df.insert(0, "review_id", range(1, len(df) + 1))

    # Add sentiment label from rating
    df["sentiment_label"] = df["rating"].apply(rating_to_sentiment)

    # Add placeholder date column
    # This dataset usually does not contain review dates.
    # Later, if we use another dataset with timestamps, we can replace this.
    df["date"] = None

    # Add placeholder issue label column
    # Later we will manually annotate a subset for supervised issue classification.
    df["issue_label"] = None

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("Cleaning finished.")
    print(f"Saved cleaned data to: {OUT_PATH}")
    print(f"Number of cleaned reviews: {len(df)}")
    print("=" * 60)
    print(df.head())


if __name__ == "__main__":
    main()