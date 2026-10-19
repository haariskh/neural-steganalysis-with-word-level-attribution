
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import shap

from lime.lime_text import LimeTextExplainer
from shap.maskers import Text
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate IG, LIME, and SHAP explanations."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Test dataset row to explain (default: 0).",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=20,
        help="Integrated Gradients steps (default: 20).",
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=300,
        help="Number of LIME perturbations (default: 300).",
    )
    parser.add_argument(
        "--shap-evals",
        type=int,
        default=100,
        help="SHAP evaluation budget (default: 100).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.ig_steps < 1 or args.lime_samples < 1 or args.shap_evals < 1:
        raise ValueError("Explanation budgets must be positive.")

    # --------------------------------------------------
    # Device and paths
    # --------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    checkpoint_dir = Path(CHECKPOINT_DIR) / "bert_detector"
    dataset_path = (
        Path(PROCESSED_DATA_DIR) / "test_binary_detector.csv"
    )
    output_dir = Path(EXPLANATION_DIR)

    ig_dir = output_dir / "integrated_gradients"
    lime_dir = output_dir / "lime"
    shap_dir = output_dir / "shap"

    for directory in (ig_dir, lime_dir, shap_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"BERT checkpoint not found: {checkpoint_dir}"
        )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {dataset_path}"
        )

    # --------------------------------------------------
    # Load one held-out review
    # --------------------------------------------------
    print("Loading test dataset...")
    dataframe = pd.read_csv(dataset_path)

    if not 0 <= args.index < len(dataframe):
        raise IndexError(
            f"Invalid index {args.index}; valid range is "
            f"0 to {len(dataframe) - 1}."
        )

    row = dataframe.iloc[args.index]
    text = str(row["review"])
    actual_label = int(row["is_stego"])

    if actual_label not in (0, 1):
        raise ValueError(f"Unexpected label: {actual_label}")

    class_names = ["cover", "synonym-stego"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    print("Loading trained BERT detector...")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=2,
    )
    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Shared prediction function
    # Returns probabilities in [cover, synonym-stego] order.
    # --------------------------------------------------
    def predict_proba(texts):
        if isinstance(texts, str):
            texts = [texts]

        all_probabilities = []
        batch_size = 4

        for start in range(0, len(texts), batch_size):
            batch = [str(item) for item in texts[start:start + batch_size]]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)

            all_probabilities.extend(
                probabilities.detach().cpu().numpy()
            )

        return np.asarray(all_probabilities, dtype=np.float64)

    probabilities = predict_proba([text])[0]
    predicted_label = int(np.argmax(probabilities))

    print()
    print(f"Dataset index: {args.index}")
    print(f"Actual class: {class_names[actual_label]}")
    print(f"Predicted class: {class_names[predicted_label]}")
    print(f"Cover probability: {probabilities[0]:.4f}")
    print(f"Stego probability: {probabilities[1]:.4f}")

    # --------------------------------------------------
    # 1. Integrated Gradients
    # --------------------------------------------------
    print("\nGenerating Integrated Gradients...")

    ig_explainer = IntegratedGradientsExplainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_steps=args.ig_steps,
    )

    ig_result = ig_explainer.explain(
        text=text,
        target_class=predicted_label,
    )

    ig_result["dataset_index"] = args.index
    ig_result["actual_class"] = actual_label
    ig_result["actual_class_name"] = class_names[actual_label]

    ig_json = ig_dir / f"explanation_{args.index:05d}.json"
    ig_html = ig_dir / f"explanation_{args.index:05d}.html"

    IntegratedGradientsExplainer.save_json(
        explanation=ig_result,
        output_path=ig_json,
    )
    IntegratedGradientsExplainer.save_html(
        explanation=ig_result,
        output_path=ig_html,
    )

    print(f"IG JSON: {ig_json}")
    print(f"IG HTML: {ig_html}")

    # --------------------------------------------------
    # 2. LIME
    # --------------------------------------------------
    print("\nGenerating LIME...")

    lime_explainer = LimeTextExplainer(
        class_names=class_names,
        random_state=42,
    )

    lime_result = lime_explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        labels=[0, 1],
        num_features=15,
        num_samples=args.lime_samples,
    )

    lime_data = {
        "dataset_index": args.index,
        "text": text,
        "actual_label": actual_label,
        "actual_class": class_names[actual_label],
        "predicted_label": predicted_label,
        "predicted_class": class_names[predicted_label],
        "probabilities": probabilities.tolist(),
        "class_explanations": {
            str(label): [
                {"word": word, "weight": float(weight)}
                for word, weight in lime_result.as_list(label=label)
            ]
            for label in (0, 1)
        },
    }

    lime_json = lime_dir / f"explanation_{args.index:05d}.json"
    lime_html = lime_dir / f"explanation_{args.index:05d}.html"

    lime_json.write_text(
        json.dumps(lime_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lime_result.save_to_file(str(lime_html))

    print(f"LIME JSON: {lime_json}")
    print(f"LIME HTML: {lime_html}")

    # --------------------------------------------------
    # 3. SHAP
    # --------------------------------------------------
    print("\nGenerating SHAP...")

    masker = Text(tokenizer)

    shap_explainer = shap.Explainer(
        predict_proba,
        masker,
        output_names=class_names,
    )

    shap_result = shap_explainer(
        [text],
        max_evals=args.shap_evals,
    )

    # Normalize the result in case the SHAP version returns
    # a list of Explanation objects for model outputs.
    if isinstance(shap_result, list):
        if len(shap_result) != 1:
            raise ValueError(
                f"Unexpected SHAP result list length: {len(shap_result)}"
            )
        shap_values = shap_result[0]
    else:
        shap_values = shap_result

    raw_values = np.asarray(shap_values.values)
    raw_data = shap_values.data

    # Expected for this probability function:
    # (samples, tokens, classes)
    if raw_values.ndim != 3 or raw_values.shape[0] != 1:
        raise ValueError(
            f"Unexpected SHAP dimensions: {raw_values.shape}. "
            "Expected (1, tokens, 2)."
        )

    values = raw_values[0]
    tokens = [str(token).strip() for token in raw_data[0]]

    if values.shape[0] != len(tokens):
        raise ValueError(
            f"SHAP token/value mismatch: {len(tokens)} tokens "
            f"but {values.shape[0]} attribution rows."
        )

    if values.shape[1] != 2:
        raise ValueError(
            f"Expected two SHAP outputs; got {values.shape[1]}."
        )

    # Explain the predicted class explicitly.
    target_values = values[:, predicted_label]

    shap_base_values = np.asarray(shap_values.base_values)

    shap_data = {
        "dataset_index": args.index,
        "text": text,
        "actual_label": actual_label,
        "actual_class": class_names[actual_label],
        "predicted_label": predicted_label,
        "predicted_class": class_names[predicted_label],
        "probabilities": probabilities.tolist(),
        "explained_class": class_names[predicted_label],
        "tokens": tokens,
        "target_attributions": target_values.tolist(),
        "all_class_attributions": raw_values.tolist(),
        "base_values": shap_base_values.tolist(),
    }

    shap_json = shap_dir / f"explanation_{args.index:05d}.json"
    shap_html = shap_dir / f"explanation_{args.index:05d}.html"

    shap_json.write_text(
        json.dumps(shap_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    top_indices = np.argsort(
        np.abs(target_values)
    )[-20:][::-1]

    rows = []
    for token_index in top_indices:
        token = html.escape(tokens[token_index])
        score = float(target_values[token_index])
        direction = (
            "toward explained class"
            if score > 0
            else "away from explained class"
            if score < 0
            else "neutral"
        )
        rows.append(
            f"<tr><td>{token}</td>"
            f"<td>{score:.6f}</td>"
            f"<td>{direction}</td></tr>"
        )

    html_content = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SHAP Explanation</title></head><body>"
        "<h2>SHAP Token Attribution</h2>"
        f"<p>Actual class: {html.escape(class_names[actual_label])}</p>"
        f"<p>Predicted class: {html.escape(class_names[predicted_label])}</p>"
        f"<p>Explained output: {html.escape(class_names[predicted_label])}</p>"
        f"<p>Cover probability: {probabilities[0]:.6f}</p>"
        f"<p>Stego probability: {probabilities[1]:.6f}</p>"
        "<p>Positive values support the explained output; "
        "negative values oppose it. These are attributions, "
        "not proof of causality.</p>"
        "<table border='1' cellpadding='6'>"
        "<tr><th>Token</th><th>SHAP value</th><th>Direction</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )

    shap_html.write_text(html_content, encoding="utf-8")

    print(f"SHAP JSON: {shap_json}")
    print(f"SHAP HTML: {shap_html}")

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    summary = {
        "dataset_index": args.index,
        "actual_class": class_names[actual_label],
        "predicted_class": class_names[predicted_label],
        "probabilities": {
            "cover": float(probabilities[0]),
            "synonym_stego": float(probabilities[1]),
        },
        "methods_generated": [
            "Integrated Gradients",
            "LIME",
            "SHAP",
        ],
        "output_directory": str(output_dir.resolve()),
    }

    summary_path = output_dir / f"comparison_summary_{args.index:05d}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nAll three explanation methods completed.")
    print(f"Summary: {summary_path}")
    print(f"Output root: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
