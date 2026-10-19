
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import MAX_LENGTH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "test_binary_detector.csv"
MODEL_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "bert_detector"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

BATCH_SIZE = 4


class TestDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.reviews = dataframe["review"].astype(str).tolist()
        self.labels = dataframe["is_stego"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.reviews[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Test dataset not found: {DATA_FILE}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        raise ValueError("The test dataset is empty.")

    if df["review"].isna().any() or df["is_stego"].isna().any():
        raise ValueError("Test data contains missing reviews or labels.")

    df["is_stego"] = df["is_stego"].astype(int)

    if not set(df["is_stego"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 (cover) or 1 (stego).")

    if df["is_stego"].nunique() != 2:
        raise ValueError("The test set must contain both classes.")

    print(f"Test reviews: {len(df)}")
    print("Class distribution:")
    print(df["is_stego"].value_counts().sort_index())

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )
    model.to(device)
    model.eval()

    dataset = TestDataset(df, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("\nEvaluating saved checkpoint...")

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probabilities = torch.softmax(outputs.logits, dim=-1)
            predictions = probabilities.argmax(dim=-1)

            all_labels.extend(batch["labels"].tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )

            if batch_index % 250 == 0:
                print(f"Processed {len(all_labels)}/{len(dataset)} reviews")

    accuracy = accuracy_score(all_labels, all_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0,
    )

    roc_auc = roc_auc_score(all_labels, all_probabilities)
    matrix = confusion_matrix(all_labels, all_predictions, labels=[0, 1])

    report = {
        "model": str(MODEL_DIR),
        "test_samples": len(all_labels),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["cover", "stego"],
            output_dict=True,
            zero_division=0,
        ),
    }

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Accuracy:  {accuracy:.4%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("                 Predicted cover  Predicted stego")
    print(f"Actual cover     {matrix[0, 0]:14d}  {matrix[0, 1]:15d}")
    print(f"Actual stego     {matrix[1, 0]:14d}  {matrix[1, 1]:15d}")

    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["cover", "stego"],
            zero_division=0,
        )
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "bert_detector_evaluation.json"
    predictions_path = REPORT_DIR / "bert_detector_predictions.csv"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    predictions_df = pd.DataFrame({
        "review": df["review"].tolist(),
        "actual_label": all_labels,
        "predicted_label": all_predictions,
        "stego_probability": all_probabilities,
        "correct": [
            actual == predicted
            for actual, predicted in zip(all_labels, all_predictions)
        ],
    })

    predictions_df.to_csv(predictions_path, index=False)

    print(f"\nMetrics saved to: {report_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
