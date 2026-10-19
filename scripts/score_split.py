"""
Score one split with the trained detector and save per-sample scores.

Kept deliberately small: this is the only step that needs a GPU. All
analysis runs on the CSV this writes, so the expensive part happens once
and the tables can be rebuilt without touching the model.

Usage:
    python -m scripts.score_split --split validation
    python -m scripts.score_split --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import CHECKPOINT_DIR, MAX_LENGTH, PROCESSED_DATA_DIR, REPORT_DIR


class _TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    checkpoint = Path(
        args.checkpoint
        if args.checkpoint
        else Path(CHECKPOINT_DIR) / "bert_detector"
    )

    dataset_path = (
        Path(PROCESSED_DATA_DIR) / f"{args.split}_binary_detector.csv"
    )

    for path in (checkpoint, dataset_path):
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"MAX_LENGTH: {MAX_LENGTH}")

    dataframe = pd.read_csv(dataset_path)

    for column in ("review", "is_stego", "original_id"):
        if column not in dataframe.columns:
            raise ValueError(f"{dataset_path} lacks column '{column}'")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2
    )
    model.to(device)
    model.eval()

    loader = DataLoader(
        _TextDataset(
            dataframe["review"].astype(str).tolist(),
            tokenizer,
            MAX_LENGTH,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    probabilities = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Scoring {args.split}"):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            probabilities.extend(
                torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
            )

    output = pd.DataFrame(
        {
            "original_id": dataframe["original_id"],
            "is_stego": dataframe["is_stego"].astype(int),
            "stego_probability": probabilities,
        }
    )

    report_dir = Path(REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"{args.split}_scores.csv"
    output.to_csv(output_path, index=False)

    print(f"\nSaved {len(output)} scores to: {output_path}")


if __name__ == "__main__":
    main()
