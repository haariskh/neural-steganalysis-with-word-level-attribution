from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import (
    CHECKPOINT_DIR,
    EXPLANATION_DIR,
    PROCESSED_DATA_DIR,
)

from src.explainability.integrated_gradients import (
    IntegratedGradientsExplainer,
)


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Integrated Gradients explanations "
            "for the trained BERT steganalysis detector."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help=(
            "Row index from the test detector dataset. "
            "Default: 0"
        ),
    )

    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        choices=[0, 1],
        help=(
            "Class to explain. "
            "0 = cover, 1 = synonym-stego. "
            "If omitted, the predicted class is explained."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help=(
            "Number of Integrated Gradients interpolation "
            "steps. Default: 20"
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# Main function
# --------------------------------------------------

def main() -> None:
    args = parse_args()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Project paths
    # --------------------------------------------------

    checkpoint_dir = (
        Path(CHECKPOINT_DIR)
        / "bert_detector"
    )

    dataset_path = (
        Path(PROCESSED_DATA_DIR)
        / "test_binary_detector.csv"
    )

    output_dir = (
        Path(EXPLANATION_DIR)
        / "integrated_gradients"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Validate paths
    # --------------------------------------------------

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Trained BERT checkpoint not found:\n"
            f"{checkpoint_dir}"
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Test detector dataset not found:\n"
            f"{dataset_path}"
        )

    # --------------------------------------------------
    # Load test dataset
    # --------------------------------------------------

    print("Loading test dataset...")

    dataframe = pd.read_csv(dataset_path)

    if args.index < 0 or args.index >= len(dataframe):
        raise IndexError(
            f"Invalid index {args.index}. "
            f"Valid range: 0 to {len(dataframe) - 1}."
        )

    row = dataframe.iloc[args.index]

    text = str(row["review"])
    actual_label = int(row["is_stego"])

    actual_label_name = (
        "synonym-stego"
        if actual_label == 1
        else "cover"
    )

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir
    )

    # --------------------------------------------------
    # Load trained BERT detector
    # --------------------------------------------------

    print("Loading trained BERT detector...")

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=2,
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Create Integrated Gradients explainer
    # --------------------------------------------------

    explainer = IntegratedGradientsExplainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_steps=args.steps,
    )

    # --------------------------------------------------
    # Generate explanation
    # --------------------------------------------------

    print("Generating Integrated Gradients explanation...")
    print(f"Dataset index: {args.index}")
    print(f"Actual label: {actual_label}")
    print(f"Actual class: {actual_label_name}")
    print(f"Text length: {len(text)} characters")

    explanation = explainer.explain(
        text=text,
        target_class=args.target_class,
    )

    # Add dataset information to the explanation
    explanation["dataset_index"] = args.index
    explanation["actual_class"] = actual_label
    explanation["actual_class_name"] = actual_label_name

    # --------------------------------------------------
    # Output paths
    # --------------------------------------------------

    json_path = (
        output_dir
        / f"explanation_{args.index:05d}.json"
    )

    html_path = (
        output_dir
        / f"explanation_{args.index:05d}.html"
    )

    # --------------------------------------------------
    # Save explanation
    # --------------------------------------------------

    IntegratedGradientsExplainer.save_json(
        explanation=explanation,
        output_path=json_path,
    )

    IntegratedGradientsExplainer.save_html(
        explanation=explanation,
        output_path=html_path,
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print()
    print("Explanation generated successfully.")

    print(
        f"Actual class: "
        f"{explanation['actual_class_name']}"
    )

    print(
        f"Predicted class: "
        f"{explanation['predicted_class_name']}"
    )

    print(
        f"Cover probability: "
        f"{explanation['cover_probability']:.4f}"
    )

    print(
        f"Stego probability: "
        f"{explanation['stego_probability']:.4f}"
    )

    print()
    print(f"JSON saved to: {json_path}")
    print(f"HTML saved to: {html_path}")

    # --------------------------------------------------
    # Display most influential words
    # --------------------------------------------------

    print()
    print("Top influential words:")

    sorted_words = sorted(
        explanation["words"],
        key=lambda item: abs(
            float(item["attribution"])
        ),
        reverse=True,
    )

    for item in sorted_words[:15]:
        print(
            f"{item['word']:20s} "
            f"{float(item['attribution']): .6f}"
        )


if __name__ == "__main__":
    main()  