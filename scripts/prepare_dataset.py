
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, create_directories


def load_imdb_dataset() -> pd.DataFrame:
    """Load, validate, and deduplicate the original IMDB reviews."""

    file_path = RAW_DATA_DIR / "IMDB Dataset.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"IMDB dataset not found at: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {"review", "sentiment"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    # Remove missing or empty reviews.
    df["review"] = df["review"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    df = df[df["review"].str.strip().ne("")].copy()

    # Encode sentiment: positive = 1, negative = 0.
    df["sentiment_label"] = df["sentiment"].map(
        {"positive": 1, "negative": 0}
    )

    if df["sentiment_label"].isna().any():
        raise ValueError("Unexpected or missing sentiment labels.")

    print(f"Original valid rows: {len(df)}")

    # Normalize whitespace and case to detect duplicate review text.
    df["_review_key"] = (
        df["review"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Do not silently discard reviews with conflicting sentiment labels.
    label_counts = df.groupby("_review_key")[
        "sentiment_label"
    ].nunique()

    conflicting_keys = label_counts[label_counts > 1]

    if not conflicting_keys.empty:
        examples = df[
            df["_review_key"].isin(conflicting_keys.index)
        ][["_review_key", "sentiment"]].head(10)

        raise ValueError(
            f"Found {len(conflicting_keys)} review texts with conflicting "
            "sentiment labels. Inspect these examples before proceeding:\n"
            f"{examples.to_string(index=False)}"
        )

    # Keep the first copy of each review.
    before = len(df)
    df = df.drop_duplicates(
        subset="_review_key",
        keep="first",
    ).copy()

    removed = before - len(df)

    print(f"Duplicate reviews removed: {removed}")
    print(f"Unique reviews remaining: {len(df)}")

    df = df.drop(columns=["_review_key"]).reset_index(drop=True)

    # Assign IDs only after deduplication.
    df.insert(0, "original_id", range(len(df)))

    return df[
        [
            "original_id",
            "review",
            "sentiment",
            "sentiment_label",
        ]
    ]


def split_dataset(df: pd.DataFrame):
    """Create stratified 70% / 15% / 15% splits."""

    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["sentiment_label"],
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=42,
        stratify=temporary_df["sentiment_label"],
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(train_df, validation_df, test_df):
    """Save the deduplicated original review splits."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for name, df in splits.items():
        output_file = PROCESSED_DATA_DIR / f"{name}_original.csv"
        df.to_csv(output_file, index=False)

        print(
            f"Saved {name}: {len(df)} rows -> {output_file}"
        )


def main():
    create_directories()

    print("Loading and deduplicating IMDB dataset...")
    df = load_imdb_dataset()

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    train_df, validation_df, test_df = split_dataset(df)

    print("\nSplit sizes:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:       {len(test_df)}")

    save_splits(train_df, validation_df, test_df)

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()
