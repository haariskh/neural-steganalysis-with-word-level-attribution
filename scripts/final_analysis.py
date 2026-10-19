"""
Build every quantitative table the write-up needs, from saved scores.

Runs on CPU against the CSVs written by scripts/score_split.py, so the
tables can be regenerated instantly without re-running the model.

Produces:
  1. Headline test metrics at the 0.5 threshold.
  2. Test metrics at a threshold chosen on VALIDATION. Choosing a
     threshold on the test split and reporting on that same split leaks
     the labels; this keeps the two separate.
  3. Bootstrap confidence intervals, so a single figure is not mistaken
     for a precise one.
  4. The accuracy ceiling implied by ROC-AUC, which says whether the
     decision threshold or the signal itself is the binding constraint.
  5. Detection as a function of embedding rate, by stratifying the test
     split on the encoder's recorded num_substitutions.

Table 5 is the one worth leading with: accuracy quoted without an
embedding rate is not interpretable for a steganalysis detector.

Usage:
    python -m scripts.final_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config import GENERATED_DATA_DIR, REPORT_DIR
from src.evaluation.metrics import (
    accuracy_ceiling_from_auc,
    auc_required_for_accuracy,
    bootstrap_confidence_interval,
    compute_metrics,
    select_threshold,
)


def load_scores(split: str, report_dir: Path) -> pd.DataFrame:
    path = report_dir / f"{split}_scores.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}\nRun: python -m scripts.score_split --split {split}"
        )
    return pd.read_csv(path)


def substitution_counts(split: str, generated_dir: Path) -> Dict[int, int]:
    """original_id -> number of substitutions the encoder actually made."""
    path = generated_dir / f"{split}_synonym_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {path}")

    records = json.loads(path.read_text(encoding="utf-8"))

    counts: Dict[int, int] = {}
    missing = 0

    for record in records:
        if "original_id" not in record:
            missing += 1
            continue
        counts[int(record["original_id"])] = int(
            record.get(
                "num_substitutions",
                len(record.get("substitutions", [])),
            )
        )

    if missing:
        raise ValueError(
            f"{missing} metadata records lack original_id. Run: "
            "python -m scripts.repair_metadata_ids"
        )

    return counts


def rate_table(
    scores: pd.DataFrame,
    counts: Dict[int, int],
    threshold: float,
    bins: List[tuple],
) -> List[Dict[str, Any]]:
    """
    Detection performance per embedding-rate stratum.

    Each stratum pairs the stego rows in that substitution range against
    ALL cover rows, so every row is a proper binary problem with the same
    negative class. The strata share one test split and one detector
    trained on the mixed distribution, so they are not independent
    experiments; they describe the detector's operating range, which is
    what a reader wants to know.
    """

    cover = scores[scores["is_stego"] == 0]
    stego = scores[scores["is_stego"] == 1].copy()
    stego["n_subs"] = stego["original_id"].map(counts)

    unmapped = int(stego["n_subs"].isna().sum())
    stego = stego.dropna(subset=["n_subs"])
    stego["n_subs"] = stego["n_subs"].astype(int)

    rows: List[Dict[str, Any]] = []

    for low, high in bins:
        subset = stego[(stego["n_subs"] >= low) & (stego["n_subs"] <= high)]
        if len(subset) < 20:
            continue

        combined = pd.concat([cover, subset], ignore_index=True)
        metrics = compute_metrics(
            combined["is_stego"].values,
            combined["stego_probability"].values,
            threshold=threshold,
        )

        # Raw accuracy is not comparable across strata: every stratum is
        # scored against the SAME full cover set, so a small stego
        # stratum is swamped by true negatives. balanced_accuracy and
        # recall are the interpretable columns here.
        rows.append(
            {
                "substitutions": f"{low}" if low == high else f"{low}-{high}",
                "n_stego": len(subset),
                "mean_subs": float(subset["n_subs"].mean()),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "recall": metrics["recall"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    if unmapped:
        print(f"  note: {unmapped} stego rows had no metadata match")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--criterion", default="balanced_accuracy")
    args = parser.parse_args()

    report_dir = Path(
        args.report_dir if args.report_dir else REPORT_DIR
    )

    validation = load_scores("validation", report_dir)
    test = load_scores("test", report_dir)

    y_test = test["is_stego"].values
    s_test = test["stego_probability"].values

    # --- 1. default threshold -------------------------------------
    default = compute_metrics(y_test, s_test, 0.5)

    # --- 2. threshold chosen on validation ------------------------
    chosen, criterion_value = select_threshold(
        validation["is_stego"].values,
        validation["stego_probability"].values,
        criterion=args.criterion,
    )
    tuned = compute_metrics(y_test, s_test, chosen)

    # --- 3. confidence intervals ----------------------------------
    ci_accuracy = bootstrap_confidence_interval(
        y_test, s_test, "accuracy", 0.5, n_resamples=1000
    )
    ci_auc = bootstrap_confidence_interval(
        y_test, s_test, "roc_auc", 0.5, n_resamples=1000
    )

    # --- 4. ceiling -----------------------------------------------
    ceiling = accuracy_ceiling_from_auc(default["roc_auc"])

    # --- 5. embedding-rate curve ----------------------------------
    counts = substitution_counts("test", Path(GENERATED_DATA_DIR))
    bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 8), (9, 99)]
    rates = rate_table(test, counts, 0.5, bins)

    # ---------------- output ----------------
    print("\n" + "=" * 62)
    print("TABLE 1  Test metrics at threshold 0.5")
    print("=" * 62)
    for key in (
        "accuracy", "balanced_accuracy", "precision",
        "recall", "specificity", "f1", "roc_auc", "average_precision",
    ):
        print(f"  {key:20} {default[key]:.4f}")
    print(f"  {'n_samples':20} {default['n_samples']}")

    print("\n" + "=" * 62)
    print(f"TABLE 2  Threshold selected on VALIDATION ({args.criterion})")
    print("=" * 62)
    print(f"  chosen threshold     {chosen:.4f}")
    print(f"  {'accuracy':20} {tuned['accuracy']:.4f}"
          f"   (0.5 gave {default['accuracy']:.4f})")
    print(f"  {'precision':20} {tuned['precision']:.4f}")
    print(f"  {'recall':20} {tuned['recall']:.4f}")
    print(f"  {'f1':20} {tuned['f1']:.4f}")

    print("\n" + "=" * 62)
    print("TABLE 3  95% bootstrap confidence intervals")
    print("=" * 62)
    print(f"  accuracy  {ci_accuracy['point_estimate']:.4f}  "
          f"[{ci_accuracy['lower']:.4f}, {ci_accuracy['upper']:.4f}]")
    print(f"  roc_auc   {ci_auc['point_estimate']:.4f}  "
          f"[{ci_auc['lower']:.4f}, {ci_auc['upper']:.4f}]")

    print("\n" + "=" * 62)
    print("TABLE 4  Separability and the accuracy ceiling")
    print("=" * 62)
    print(f"  ROC-AUC                        {ceiling['roc_auc']:.4f}")
    print(f"  separability d'                {ceiling['separability_d_prime']:.3f}")
    print(f"  max accuracy at best threshold {ceiling['estimated_max_accuracy']:.4f}")
    print(f"  observed accuracy              {default['accuracy']:.4f}")
    print(f"  AUC required for 95% accuracy  {auc_required_for_accuracy(0.95):.4f}")
    gap = default["accuracy"] - ceiling["estimated_max_accuracy"]
    print(
        "  -> accuracy is at the ceiling separability allows; "
        "only a stronger signal raises it."
        if gap > -0.01
        else "  -> threshold tuning has room; see Table 2."
    )

    print("\n" + "=" * 62)
    print("TABLE 5  Detection vs embedding rate (stego stratum vs all cover)")
    print("=" * 62)
    print(f"  {'subs':>6} {'n':>6} {'mean':>6} {'bal.acc':>9} {'recall':>8} {'AUC':>8}")
    print("  " + "-" * 48)
    for row in rates:
        print(
            f"  {row['substitutions']:>6} {row['n_stego']:>6} "
            f"{row['mean_subs']:>6.1f} {row['balanced_accuracy']:>9.4f} "
            f"{row['recall']:>8.4f} {row['roc_auc']:>8.4f}"
        )
    print("\n  Detection improves with embedding rate. Quote accuracy WITH")
    print("  the rate it was measured at; a bare figure is not comparable.")

    out = {
        "threshold_0.5": default,
        "threshold_selected_on_validation": {
            "criterion": args.criterion,
            "threshold": chosen,
            "validation_criterion_value": criterion_value,
            "test_metrics": tuned,
        },
        "confidence_intervals": {
            "accuracy": ci_accuracy,
            "roc_auc": ci_auc,
        },
        "ceiling_analysis": ceiling,
        "auc_required_for_95_accuracy": auc_required_for_accuracy(0.95),
        "embedding_rate_curve": rates,
    }

    path = report_dir / "final_analysis.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    main()
