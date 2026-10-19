from pathlib import Path
from typing import Tuple

import pandas as pd


def load_original_split(
    processed_dir: Path,
    split: str,
) -> pd.DataFrame:
    """
    Loads original cover reviews for a given split.
    """

    file_path = processed_dir / f"{split}_original.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Original split not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "original_id",
        "review",
        "sentiment",
        "sentiment_label",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path}: {missing_columns}"
        )

    df = df.copy()

    df["algorithm"] = "cover"
    df["algorithm_label"] = 0
    df["is_stego"] = 0

    return df


def load_synonym_split(
    generated_dir: Path,
    split: str,
) -> pd.DataFrame:
    """
    Loads synonym-stego reviews for a given split.
    """

    file_path = generated_dir / f"{split}_synonym.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Synonym split not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "original_id",
        "review",
        "sentiment",
        "sentiment_label",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path}: {missing_columns}"
        )

    df = df.copy()

    df["algorithm"] = "synonym"
    df["algorithm_label"] = 1
    df["is_stego"] = 1

    return df


def build_binary_detector_dataset(
    processed_dir: Path,
    generated_dir: Path,
    split: str,
) -> pd.DataFrame:
    """
    Combines cover and synonym-stego reviews for binary detection.

    Labels:
        0 = cover
        1 = stego
    """

    cover_df = load_original_split(
        processed_dir=processed_dir,
        split=split,
    )

    synonym_df = load_synonym_split(
        generated_dir=generated_dir,
        split=split,
    )

    # Keep only the columns needed by the detector.
    columns = [
        "original_id",
        "review",
        "sentiment",
        "sentiment_label",
        "algorithm",
        "algorithm_label",
        "is_stego",
    ]

    cover_df = cover_df[columns]
    synonym_df = synonym_df[columns]

    combined_df = pd.concat(
        [cover_df, synonym_df],
        ignore_index=True,
    )

    # Shuffle deterministically.
    combined_df = combined_df.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)

    return combined_df


def save_binary_detector_dataset(
    processed_dir: Path,
    generated_dir: Path,
    output_dir: Path,
    split: str,
) -> Path:
    """
    Builds and saves a binary detector dataset.
    """

    df = build_binary_detector_dataset(
        processed_dir=processed_dir,
        generated_dir=generated_dir,
        split=split,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / f"{split}_binary_detector.csv"

    df.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved: {output_file}")
    print(f"Rows: {len(df)}")
    print(df["is_stego"].value_counts().sort_index())

    return output_file