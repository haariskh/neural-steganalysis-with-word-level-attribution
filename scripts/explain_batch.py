"""
Generate IG, LIME and SHAP explanations for many test rows in one process.

scripts/explain_all.py handles a single row and reloads BERT each time it
is invoked. Looping it over a hundred indices would spend most of the
runtime loading the model, so this script loads once and iterates.

Output filenames and JSON schemas are identical to explain_all.py, so
scripts/evaluate_localization.py consumes either interchangeably.

Only stego rows can be scored for localization: a cover review has no
substituted words, so "did the explainer find them" is not a question
that can be asked of it. --only-stego is therefore the default.

Usage:
    python -m scripts.explain_batch --limit 100
    python -m scripts.explain_batch --limit 100 --only-correct
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_text import LimeTextExplainer
from shap.maskers import Text
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    CHECKPOINT_DIR,
    EXPLANATION_DIR,
    MAX_LENGTH,
    PROCESSED_DATA_DIR,
)
from src.explainability.integrated_gradients import (
    IntegratedGradientsExplainer,
)

CLASS_NAMES = ["cover", "synonym-stego"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help=(
            "TOTAL rows that should end up explained, counting rows "
            "already done in an earlier run. --limit 150 with 60 already "
            "present generates 90 more, not 150 more."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--ig-steps", type=int, default=20)
    parser.add_argument("--lime-samples", type=int, default=300)
    parser.add_argument("--shap-evals", type=int, default=100)
    parser.add_argument(
        "--only-stego",
        action="store_true",
        default=True,
        help="Explain stego rows only (default).",
    )
    parser.add_argument(
        "--include-cover",
        dest="only_stego",
        action="store_false",
        help="Also explain cover rows.",
    )
    parser.add_argument(
        "--only-correct",
        action="store_true",
        help="Explain only rows the detector classified correctly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate rows that already have all three files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = Path(CHECKPOINT_DIR) / "bert_detector"
    dataset_path = (
        Path(PROCESSED_DATA_DIR) / f"{args.split}_binary_detector.csv"
    )

    for path in (checkpoint, dataset_path):
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

    root = Path(EXPLANATION_DIR)
    directories = {
        "ig": root / "integrated_gradients",
        "lime": root / "lime",
        "shap": root / "shap",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"MAX_LENGTH: {MAX_LENGTH}")

    dataframe = pd.read_csv(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2
    )
    model.to(device)
    model.eval()

    def predict_proba(texts):
        if isinstance(texts, str):
            texts = [texts]
        output = []
        for start in range(0, len(texts), 8):
            batch = [str(t) for t in texts[start:start + 8]]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(
                    model(**inputs).logits, dim=-1
                )
            output.extend(probabilities.detach().cpu().numpy())
        return np.asarray(output, dtype=np.float64)

    ig_explainer = IntegratedGradientsExplainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_steps=args.ig_steps,
    )
    lime_explainer = LimeTextExplainer(
        class_names=CLASS_NAMES, random_state=42
    )
    shap_explainer = shap.Explainer(
        predict_proba, Text(tokenizer), output_names=CLASS_NAMES
    )

    # Choose which rows to explain.
    candidates = []
    for index, row in dataframe.iterrows():
        if index < args.start:
            continue
        if args.only_stego and int(row["is_stego"]) != 1:
            continue
        candidates.append(int(index))

    done = 0          # newly generated in this run
    already = 0       # complete from an earlier run
    skipped = 0       # filtered out
    failed = []
    started = time.time()

    for index in candidates:
        # --limit is a TARGET TOTAL, so rows carried over from an earlier
        # run count toward it. Otherwise resuming a batch silently
        # generates another full --limit on top of what exists.
        if done + already >= args.limit:
            break

        paths = {
            key: directory / f"explanation_{index:05d}.json"
            for key, directory in directories.items()
        }

        if not args.overwrite and all(p.exists() for p in paths.values()):
            already += 1
            continue

        row = dataframe.iloc[index]
        text = str(row["review"])
        actual_label = int(row["is_stego"])

        try:
            probabilities = predict_proba([text])[0]
            predicted_label = int(np.argmax(probabilities))

            if args.only_correct and predicted_label != actual_label:
                skipped += 1
                continue

            # ---- Integrated Gradients ----
            ig_result = ig_explainer.explain(
                text=text, target_class=predicted_label
            )
            ig_result["dataset_index"] = index
            ig_result["actual_class"] = actual_label
            ig_result["actual_class_name"] = CLASS_NAMES[actual_label]
            paths["ig"].write_text(
                json.dumps(ig_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # ---- LIME ----
            lime_result = lime_explainer.explain_instance(
                text_instance=text,
                classifier_fn=predict_proba,
                labels=[0, 1],
                num_features=15,
                num_samples=args.lime_samples,
            )
            paths["lime"].write_text(
                json.dumps(
                    {
                        "dataset_index": index,
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "probabilities": probabilities.tolist(),
                        "class_explanations": {
                            str(label): [
                                {"word": w, "weight": float(v)}
                                for w, v in lime_result.as_list(label=label)
                            ]
                            for label in (0, 1)
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            # ---- SHAP ----
            shap_result = shap_explainer(
                [text], max_evals=args.shap_evals
            )
            shap_values = (
                shap_result[0]
                if isinstance(shap_result, list)
                else shap_result
            )
            raw_values = np.asarray(shap_values.values)
            if raw_values.ndim != 3 or raw_values.shape[0] != 1:
                raise ValueError(
                    f"Unexpected SHAP shape {raw_values.shape}"
                )
            values = raw_values[0]
            tokens = [str(t).strip() for t in shap_values.data[0]]
            if values.shape[0] != len(tokens):
                raise ValueError("SHAP token/value mismatch")

            paths["shap"].write_text(
                json.dumps(
                    {
                        "dataset_index": index,
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "probabilities": probabilities.tolist(),
                        "tokens": tokens,
                        "target_attributions": values[
                            :, predicted_label
                        ].tolist(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            done += 1

            elapsed = time.time() - started
            per_row = elapsed / done
            target = min(args.limit, len(candidates))
            remaining = max(target - (done + already), 0)
            print(
                f"[{done + already}/{target}] row {index} | "
                f"{per_row:.1f}s/row | ~{per_row * remaining / 60:.1f} min left",
                flush=True,
            )

        except Exception as error:  # noqa: BLE001
            # One bad row must not lose the whole batch.
            failed.append((index, repr(error)))
            print(f"  row {index} FAILED: {error}", flush=True)
            for path in paths.values():
                if path.exists():
                    path.unlink()

    print(
        f"\nGenerated {done} new rows in "
        f"{(time.time()-started)/60:.1f} min"
    )
    print(f"Already present from an earlier run: {already}")
    print(f"Total now explained: {done + already}")
    if skipped:
        print(f"Filtered out (misclassified): {skipped}")
    if failed:
        print(f"Failed: {len(failed)}")
        for index, error in failed[:5]:
            print(f"  {index}: {error[:120]}")


if __name__ == "__main__":
    main()
