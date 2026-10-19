"""
Back-fill `original_id` into substitution metadata generated before that
field existed.

Why this exists
---------------
scripts/generate_synonym_stega.py now records original_id so that
word-level attributions can be joined to the substitution ground truth.
Metadata written before that change has no such key, and regenerating it
means re-running the sentence-transformer encoder over the whole corpus.

The old metadata does carry `original_text`, and the cover splits carry
both `review` and `original_id`, so the id can be recovered by exact text
match. Reviews were deduplicated in prepare_dataset.py, so the mapping is
one-to-one; this script verifies that rather than assuming it, and
refuses to write anything if the match is not exact and unambiguous.

Usage:
    python -m scripts.repair_metadata_ids            # all splits
    python -m scripts.repair_metadata_ids --split test
    python -m scripts.repair_metadata_ids --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.config import GENERATED_DATA_DIR, PROCESSED_DATA_DIR


def repair_split(split: str, dry_run: bool = False) -> bool:
    metadata_path = (
        Path(GENERATED_DATA_DIR) / f"{split}_synonym_metadata.json"
    )
    original_path = Path(PROCESSED_DATA_DIR) / f"{split}_original.csv"

    if not metadata_path.exists():
        print(f"  [{split}] no metadata at {metadata_path} -- skipped")
        return False

    if not original_path.exists():
        raise FileNotFoundError(f"Cover split not found: {original_path}")

    records = json.loads(metadata_path.read_text(encoding="utf-8"))

    already = sum(1 for r in records if "original_id" in r)
    if already == len(records):
        print(f"  [{split}] all {len(records)} records already have original_id")
        return False

    originals = pd.read_csv(original_path)

    for column in ("original_id", "review"):
        if column not in originals.columns:
            raise ValueError(f"{original_path} lacks column '{column}'")

    # Exact-text -> id. Reviews were deduplicated upstream, so any
    # duplicate here means the assumption no longer holds.
    duplicated = originals["review"].duplicated().sum()
    if duplicated:
        raise ValueError(
            f"{original_path} has {duplicated} duplicate review texts; "
            "the text->id mapping is not one-to-one and the repair would "
            "assign wrong ids. Regenerate the split instead."
        )

    text_to_id = dict(
        zip(originals["review"].astype(str), originals["original_id"])
    )

    matched = 0
    unmatched = 0

    for record in records:
        if "original_id" in record:
            matched += 1
            continue

        original_text = str(record.get("original_text", ""))
        found = text_to_id.get(original_text)

        if found is None:
            unmatched += 1
            continue

        record["original_id"] = int(found)
        matched += 1

    print(
        f"  [{split}] {len(records)} records | matched {matched} | "
        f"unmatched {unmatched}"
    )

    if unmatched:
        raise ValueError(
            f"[{split}] {unmatched} records could not be matched to a cover "
            "review by exact text. Nothing was written. These records cannot "
            "be scored for localization; regenerate the split."
        )

    if dry_run:
        print(f"  [{split}] dry run -- not written")
        return False

    backup = metadata_path.with_suffix(".json.pre_id_backup")
    if not backup.exists():
        shutil.copy2(metadata_path, backup)
        print(f"  [{split}] backup -> {backup.name}")

    metadata_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  [{split}] written")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Back-fill original_id into pre-existing metadata."
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    splits = (
        [args.split]
        if args.split
        else ["train", "validation", "test"]
    )

    print("Repairing substitution metadata")
    for split in splits:
        repair_split(split, dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
