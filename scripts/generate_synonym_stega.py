import argparse
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

        # original_id makes the substitution ground truth joinable to the
        # shuffled detector dataset; without it attribution cannot be scored.
        metadata["original_id"] = int(row["original_id"])
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate synonym-substitution stego text. The embedding rate "
            "is the dominant factor in how detectable the result is, so it "
            "is exposed here rather than fixed, and every run must be "
            "reported together with the rate it used."
        )
    )
    parser.add_argument(
        "--max-substitutions",
        type=int,
        default=MAX_SYNONYM_SUBSTITUTIONS,
        help=(
            "Upper bound on substitutions per review. The achieved mean is "
            "lower, because candidates failing the similarity filter are "
            "rejected."
        ),
    )
    parser.add_argument(
        "--substitution-probability",
        type=float,
        default=0.25,
        help="Probability a candidate word is considered for substitution.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.88,
        help=(
            "Minimum context similarity for a replacement to be accepted. "
            "Lowering this raises the embedding rate but degrades semantic "
            "fidelity, which is a real cost, not a free gain."
        ),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Suffix for output filenames, e.g. --tag high writes "
            "train_synonym_high.csv. Without it the default filenames are "
            "overwritten. Use a tag to keep several embedding rates side "
            "by side."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_substitutions < 1:
        raise ValueError("--max-substitutions must be at least 1.")
    if not 0.0 < args.substitution_probability <= 1.0:
        raise ValueError("--substitution-probability must be in (0, 1].")
    if not 0.0 < args.similarity_threshold < 1.0:
        raise ValueError("--similarity-threshold must be in (0, 1).")

    create_directories()
    set_seed(42)

    print(
        f"Embedding rate settings: max_substitutions="
        f"{args.max_substitutions}, probability="
        f"{args.substitution_probability}, similarity_threshold="
        f"{args.similarity_threshold}"
    )

    encoder = SynonymSteganography(
        max_substitutions=args.max_substitutions,
        substitution_probability=args.substitution_probability,
        similarity_threshold=args.similarity_threshold,
        seed=42,
    )

    suffix = f"_{args.tag}" if args.tag else ""

    split_names = args.splits

    for split_name in split_names:
        input_file = (
            PROCESSED_DATA_DIR
            / f"{split_name}_original.csv"
        )

        output_file = (
            GENERATED_DATA_DIR
            / f"{split_name}_synonym{suffix}.csv"
        )

        metadata_file = (
            GENERATED_DATA_DIR
            / f"{split_name}_synonym_metadata{suffix}.json"
        )

        generate_split(
            input_file=input_file,
            output_file=output_file,
            metadata_file=metadata_file,
            encoder=encoder,
        )


if __name__ == "__main__":
    main()