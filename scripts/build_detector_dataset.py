import argparse
from pathlib import Path

from src.config import (
    PROCESSED_DATA_DIR,
    GENERATED_DATA_DIR,
    create_directories,
)

from src.dataset import save_binary_detector_dataset


def main():
    """
    Builds cover-versus-synonym-stego datasets for all splits.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default="",
        help=(
            "Embedding-rate tag matching generate_synonym_stega.py --tag. "
            "Reads {split}_synonym_<tag>.csv and writes "
            "{split}_binary_detector_<tag>.csv, so several rates coexist."
        ),
    )
    args = parser.parse_args()

    create_directories()

    output_dir = PROCESSED_DATA_DIR

    for split in ["train", "validation", "test"]:
        print(f"\nBuilding {split} detector dataset...")

        save_binary_detector_dataset(
            processed_dir=PROCESSED_DATA_DIR,
            generated_dir=GENERATED_DATA_DIR,
            output_dir=output_dir,
            split=split,
            tag=args.tag,
        )

    print("\nAll detector datasets created successfully.")


if __name__ == "__main__":
    main()