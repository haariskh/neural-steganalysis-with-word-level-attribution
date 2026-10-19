"""
Score IG / LIME / SHAP word-level attributions against the substitution
ground truth, and report them side by side.

This consumes explanation JSON files already written by
`python -m scripts.explain_all`, so it needs no GPU and no model load.

Usage
-----
    python -m scripts.explain_all --index 0
    python -m scripts.explain_all --index 1
    ...
    python -m scripts.evaluate_localization --split test

Only stego rows are scored. A cover row has no substituted words, so
"did the explainer find them" is not a question that can be asked of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import EXPLANATION_DIR, GENERATED_DATA_DIR, PROCESSED_DATA_DIR
from src.evaluation.localization import (
    aggregate_localization,
    load_substitution_ground_truth,
    localization_metrics,
    ranked_words_from_ig,
    ranked_words_from_lime,
    ranked_words_from_shap,
)

METHODS = ("IG", "LIME", "SHAP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score attribution localization against ground truth."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Cut-offs for precision@k / recall@k.",
    )
    parser.add_argument(
        "--explanation-dir",
        default=None,
        help="Defaults to src.config.EXPLANATION_DIR.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the JSON report.",
    )
    return parser.parse_args()


def load_explanation(directory: Path, index: int) -> Dict[str, Any] | None:
    path = directory / f"explanation_{index:05d}.json"

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()

    if any(k < 1 for k in args.top_k):
        raise ValueError("--top-k values must be >= 1.")

    explanation_root = Path(
        args.explanation_dir
        if args.explanation_dir is not None
        else EXPLANATION_DIR
    )

    dataset_path = (
        Path(PROCESSED_DATA_DIR) / f"{args.split}_binary_detector.csv"
    )
    metadata_path = (
        Path(GENERATED_DATA_DIR) / f"{args.split}_synonym_metadata.json"
    )

    for path in (dataset_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    dataframe = pd.read_csv(dataset_path)

    for column in ("original_id", "is_stego", "review"):
        if column not in dataframe.columns:
            raise ValueError(
                f"{dataset_path} is missing the '{column}' column."
            )

    ground_truth = load_substitution_ground_truth(metadata_path)

    directories = {
        "IG": explanation_root / "integrated_gradients",
        "LIME": explanation_root / "lime",
        "SHAP": explanation_root / "shap",
    }

    per_method: Dict[str, List[Dict[str, Any]]] = {m: [] for m in METHODS}
    examined = 0
    no_explanation = 0

    for index, row in dataframe.iterrows():
        # Cover rows carry no substitutions; nothing to localize.
        if int(row["is_stego"]) != 1:
            continue

        original_id = int(row["original_id"])

        if original_id not in ground_truth:
            continue

        truth_words = ground_truth[original_id]["replacement_words"]
        text = str(row["review"])

        explanations = {
            method: load_explanation(directories[method], int(index))
            for method in METHODS
        }

        # Require all three so the comparison is on identical samples.
        if any(value is None for value in explanations.values()):
            no_explanation += 1
            continue

        examined += 1

        predicted_label = int(
            explanations["IG"].get("predicted_class", 1)
        )

        rankings = {
            "IG": ranked_words_from_ig(explanations["IG"]),
            "LIME": ranked_words_from_lime(
                explanations["LIME"], predicted_label
            ),
            "SHAP": ranked_words_from_shap(
                explanations["SHAP"], predicted_label
            ),
        }

        for method in METHODS:
            metrics = localization_metrics(
                rankings[method],
                truth_words,
                text=text,
                k_values=tuple(args.top_k),
            )
            metrics["dataset_index"] = int(index)
            metrics["original_id"] = original_id
            per_method[method].append(metrics)

    if examined == 0:
        raise RuntimeError(
            "No stego row had all three explanation files.\n"
            f"Looked under: {explanation_root}\n"
            "Run 'python -m scripts.explain_all --index N' for the rows "
            "you want scored, then rerun this script."
        )

    summary = {
        method: aggregate_localization(per_method[method])
        for method in METHODS
    }

    report = {
        "split": args.split,
        "stego_rows_scored": examined,
        "rows_missing_explanations": no_explanation,
        "top_k": args.top_k,
        "summary": summary,
        "per_sample": per_method,
    }

    output_path = Path(
        args.output
        if args.output is not None
        else Path(EXPLANATION_DIR) / "localization_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # ---- console table ----
    print(f"\nLocalization vs substitution ground truth ({args.split})")
    print(f"Stego rows scored: {examined}")
    if no_explanation:
        print(f"Rows skipped (missing explanations): {no_explanation}")

    header = f"{'method':<6}" + "".join(
        f"{'P@'+str(k):>9}" for k in args.top_k
    ) + f"{'MAP':>9}{'MRR':>9}{'rankAUC':>9}{'lift@'+str(args.top_k[-1]):>10}"
    print("\n" + header)
    print("-" * len(header))

    for method in METHODS:
        stats = summary[method]
        if not stats.get("n_scored"):
            print(f"{method:<6}  (no scorable samples)")
            continue

        line = f"{method:<6}"
        for k in args.top_k:
            line += f"{stats.get(f'mean_precision_at_{k}', float('nan')):>9.3f}"
        line += f"{stats.get('mean_average_precision', float('nan')):>9.3f}"
        line += f"{stats.get('mean_reciprocal_rank', float('nan')):>9.3f}"
        line += f"{stats.get('mean_ranking_auc', float('nan')):>9.3f}"
        line += (
            f"{stats.get(f'mean_precision_at_{args.top_k[-1]}_lift', float('nan')):>9.2f}x"
        )
        print(line)

    chance = summary["IG"].get("mean_chance_precision")
    if chance is not None:
        print(
            f"\nRandom-ranking precision for these reviews: {chance:.4f}"
        )
        print(
            "lift = precision@k / chance. 1.00x means no better than random."
        )

    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
