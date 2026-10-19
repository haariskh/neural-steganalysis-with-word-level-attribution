import argparse
import json
import random
from pathlib import Path
from typing import Dict, List
import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.models.bert_detector import BertSteganalysisDetector


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "bert_detector"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

MODEL_NAME = "bert-base-uncased"

# Sequence length is defined once in src/config.py so that training,
# evaluation and all explainers tokenize identically. Vary it there.
from src.config import MAX_LENGTH
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8 #bw 4 and six
LEARNING_RATE = 2e-5
EPOCHS = 3
RANDOM_SEED = 42


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class SteganalysisDataset(Dataset):
    """
    PyTorch dataset for BERT steganalysis.

    Required columns:
        review
        is_stego
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.reviews = dataframe["review"].astype(str).tolist()
        self.labels = dataframe["is_stego"].astype(int).tolist()

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = self.reviews[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_split(
    split: str,
    smoke_test: bool = False,
) -> pd.DataFrame:

    file_path = DATA_DIR / f"{split}_binary_detector.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {"review", "is_stego"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path}: {missing_columns}"
        )

    # Small subset for the smoke test.
    if smoke_test:
        sample_size = {
            "train": 256,
            "validation": 64,
            "test": 64,
        }[split]

        sample_size = min(sample_size, len(df))

        df = df.sample(
            n=sample_size,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

    return df


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(
    model,
    dataloader,
    device,
) -> Dict[str, float]:

    model.eval()

    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[float] = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc="Evaluating",
            leave=False,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs["loss"].item()

            probabilities = torch.softmax(
                outputs["logits"],
                dim=-1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=-1,
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
                .tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(
            all_labels,
            all_probabilities,
        )
    except ValueError:
        roc_auc = 0.0

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch: int,
):

    model.train()

    total_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
    )

    for step, batch in enumerate(progress_bar):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        use_amp = device.type == "cuda"

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"]
            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(scaled_loss).backward()

        if (
            (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (step + 1) == len(dataloader)
        ):
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / max(len(dataloader), 1)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Train on a small subset to verify the pipeline.",
    )

    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print("\nLoading datasets...")

    train_df = load_split(
        "train",
        smoke_test=args.smoke_test,
    )

    validation_df = load_split(
        "validation",
        smoke_test=args.smoke_test,
    )

    test_df = load_split(
        "test",
        smoke_test=args.smoke_test,
    )

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")

    print("\nClass distribution:")
    print(train_df["is_stego"].value_counts().sort_index())

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = SteganalysisDataset(
        train_df,
        tokenizer,
        MAX_LENGTH,
    )

    validation_dataset = SteganalysisDataset(
        validation_df,
        tokenizer,
        MAX_LENGTH,
    )

    test_dataset = SteganalysisDataset(
        test_df,
        tokenizer,
        MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("\nLoading BERT model...")

    model = BertSteganalysisDetector(
        model_name=MODEL_NAME,
        num_labels=2,
    )

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    total_training_steps = (
    math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    * EPOCHS
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            0.1 * total_training_steps
        ),
        num_training_steps=total_training_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best_f1 = -1.0
    training_history = []

    print("\nStarting training...")

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        validation_metrics = evaluate(
            model=model,
            dataloader=validation_loader,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation_metrics,
        }

        training_history.append(epoch_result)

        print("\nEpoch results:")
        print(json.dumps(epoch_result, indent=2))

        if validation_metrics["f1"] > best_f1:

            best_f1 = validation_metrics["f1"]

            CHECKPOINT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.model.save_pretrained(
                CHECKPOINT_DIR
            )

            tokenizer.save_pretrained(
                CHECKPOINT_DIR
            )

            print(
                f"\nBest model saved to: {CHECKPOINT_DIR}"
            )

    print("\nLoading best model for final evaluation...")

    from transformers import AutoModelForSequenceClassification

    model.model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_DIR,
        num_labels=2,
    )

    model.to(device)

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "device": str(device),
        "smoke_test": args.smoke_test,
        "training_history": training_history,
        "test_metrics": test_metrics,
    }

    report_path = REPORT_DIR / "bert_detector_report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(f"\nReport saved to: {report_path}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()