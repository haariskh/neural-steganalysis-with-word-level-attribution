import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import (
    GENERATED_DATA_DIR,
    PROCESSED_DATA_DIR,
    MAX_SYNONYM_SUBSTITUTIONS,
    create_directories,
)
from src.steganography.synonym_stega import SynonymSteganography
from src.seed import set_seed


def generate_split(
    input_file: Path,
    output_file: Path,
    metadata_file: Path,
    encoder: SynonymSteganography,
):
    df = pd.read_csv(input_file)

    generated_rows = []
    metadata_rows = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Processing {input_file.name}",
    ):
        original_text = row["review"]

        stego_text, metadata = encoder.encode(original_text)

        # Skip unchanged samples because they are not actually stego.
        if stego_text.strip() == original_text.strip():
            continue

        generated_rows.append(
            {
                "original_id": row["original_id"],
                "review": stego_text,
                "sentiment": row["sentiment"],
                "sentiment_label": row["sentiment_label"],
                "algorithm": "synonym",
                "algorithm_label": 1,
                "is_stego": 1,
            }
        )

        metadata_rows.append(metadata)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(generated_rows).to_csv(
        output_file,
        index=False,
    )

    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(
            metadata_rows,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Generated {len(generated_rows)} stego samples.")
    print(f"Saved to: {output_file}")


def main():
    create_directories()
    set_seed(42)

    encoder = SynonymSteganography(
        max_substitutions=MAX_SYNONYM_SUBSTITUTIONS,
        substitution_probability=0.25,
        seed=42,
    )

    split_names = [
        "train",
        "validation",
        "test",
    ]

    for split_name in split_names:
        input_file = (
            PROCESSED_DATA_DIR
            / f"{split_name}_original.csv"
        )

        output_file = (
            GENERATED_DATA_DIR
            / f"{split_name}_synonym.csv"
        )

        metadata_file = (
            GENERATED_DATA_DIR
            / f"{split_name}_synonym_metadata.json"
        )

        generate_split(
            input_file=input_file,
            output_file=output_file,
            metadata_file=metadata_file,
            encoder=encoder,
        )


if __name__ == "__main__":
    main()