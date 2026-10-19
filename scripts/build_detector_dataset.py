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

    create_directories()

    output_dir = PROCESSED_DATA_DIR

    for split in ["train", "validation", "test"]:
        print(f"\nBuilding {split} detector dataset...")

        save_binary_detector_dataset(
            processed_dir=PROCESSED_DATA_DIR,
            generated_dir=GENERATED_DATA_DIR,
            output_dir=output_dir,
            split=split,
        )

    print("\nAll detector datasets created successfully.")


if __name__ == "__main__":
    main()