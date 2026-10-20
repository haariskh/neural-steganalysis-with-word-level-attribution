"""
Benchmark attribution methods against ground truth AND against the proxy
metric the explainability literature normally has to rely on.

Why this experiment exists
--------------------------
Attribution methods are usually evaluated with a deletion test: mask the
words an explainer ranked highest, and measure how far the model's
confidence falls. A large drop is taken as evidence the explanation was
faithful. The field uses this because the honest question -- "are these
the words that actually mattered?" -- normally has no answer: there is no
key saying which words carry the signal.

Steganalysis is an exception. The encoder recorded every word it
substituted, so the key exists. That makes it possible to run both
evaluations on the same samples and ask whether the proxy agrees with the
truth:

  * ground-truth localization -- does the ranking put substituted words
    first? (precision@k, MAP, rank-AUC, from src.evaluation.localization)
  * deletion test -- does masking the top-k words reduce P(stego)?

and then to correlate them per sample.

The deletion test is additionally anchored at both ends, which a
deletion test normally cannot be:

  * ORACLE  -- mask the words the encoder actually substituted. This is
    the largest drop any word-level explanation could justify.
  * RANDOM  -- mask k words drawn at random from the review. This is what
    a meaningless explanation achieves, and it is not zero: masking any
    word perturbs the input.

A method's deletion score is only interpretable between those two
anchors. Reported alone it says very little, which is part of the point.

Caveat: masking inserts [MASK] tokens the model never saw in training, so
all deletion numbers are measured off-distribution. That is a known
weakness of the metric and a further reason to treat it as a proxy rather
than as truth.

Usage:
    python -m scripts.explainer_benchmark --top-k 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    CHECKPOINT_DIR,
    EXPLANATION_DIR,
    GENERATED_DATA_DIR,
    MAX_LENGTH,
    PROCESSED_DATA_DIR,
)
from src.evaluation.localization import (
    aggregate_localization,
    load_substitution_ground_truth,
    localization_metrics,
    normalize_word,
    ranked_words_from_ig,
    ranked_words_from_lime,
    ranked_words_from_shap,
)

METHODS = ("IG", "LIME", "SHAP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Words masked in the deletion test.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mask_words(text: str, words: List[str], mask_token: str) -> str:
    """Replace whole-word occurrences of `words` with the mask token."""
    result = text
    for word in words:
        if not word:
            continue
        pattern = re.compile(
            rf"(?<!\w){re.escape(word)}(?!\w)", flags=re.IGNORECASE
        )
        result = pattern.sub(f" {mask_token} ", result)
    return result


def main() -> None:
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    rng = random.Random(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = Path(CHECKPOINT_DIR) / "bert_detector"
    dataset_path = (
        Path(PROCESSED_DATA_DIR) / f"{args.split}_binary_detector.csv"
    )
    metadata_path = (
        Path(GENERATED_DATA_DIR) / f"{args.split}_synonym_metadata.json"
    )

    for path in (checkpoint, dataset_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

    print(f"Device: {device}")

    dataframe = pd.read_csv(dataset_path)
    ground_truth = load_substitution_ground_truth(metadata_path)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2
    )
    model.to(device)
    model.eval()

    mask_token = tokenizer.mask_token or "[MASK]"

    def stego_probability(texts: List[str]) -> np.ndarray:
        output = []
        for start in range(0, len(texts), args.batch_size):
            batch = [str(t) for t in texts[start:start + args.batch_size]]
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
            output.extend(probabilities[:, 1].cpu().numpy().tolist())
        return np.asarray(output, dtype=float)

    root = Path(EXPLANATION_DIR)
    directories = {
        "IG": root / "integrated_gradients",
        "LIME": root / "lime",
        "SHAP": root / "shap",
    }

    per_method_localization: Dict[str, List[Dict[str, Any]]] = {
        m: [] for m in METHODS
    }
    rows: List[Dict[str, Any]] = []
    examined = 0

    for index, row in dataframe.iterrows():
        if int(row["is_stego"]) != 1:
            continue

        original_id = int(row["original_id"])
        if original_id not in ground_truth:
            continue

        files = {
            m: directories[m] / f"explanation_{int(index):05d}.json"
            for m in METHODS
        }
        if not all(p.exists() for p in files.values()):
            continue

        loaded = {
            m: json.loads(files[m].read_text(encoding="utf-8"))
            for m in METHODS
        }

        text = str(row["review"])
        truth_words = ground_truth[original_id]["replacement_words"]
        if not truth_words:
            continue

        predicted_label = int(loaded["IG"].get("predicted_class", 1))

        rankings = {
            "IG": ranked_words_from_ig(loaded["IG"]),
            "LIME": ranked_words_from_lime(loaded["LIME"], predicted_label),
            "SHAP": ranked_words_from_shap(loaded["SHAP"], predicted_label),
        }

        # ---- ground-truth localization ----
        for method in METHODS:
            metrics = localization_metrics(
                rankings[method],
                truth_words,
                text=text,
                k_values=(1, 3, 5, 10),
            )
            metrics["dataset_index"] = int(index)
            metrics["original_id"] = original_id
            metrics["num_substitutions"] = len(truth_words)
            per_method_localization[method].append(metrics)

        # ---- deletion test, with both anchors ----
        pool = sorted(
            {
                normalize_word(t)
                for t in re.findall(r"\w+", text)
                if len(normalize_word(t)) >= 2
            }
            - truth_words
        )
        random_words = (
            rng.sample(pool, min(args.top_k, len(pool))) if pool else []
        )

        # The oracle masks up to k of the ACTUAL substituted words, not all
        # of them: it stands for a perfect top-k explanation, so it must
        # mask the same number of words the methods do. Masking every
        # substitution would perturb the text more and make the anchor
        # unreachable by construction rather than by quality.
        truth_sorted = sorted(truth_words)
        oracle_words = (
            rng.sample(truth_sorted, args.top_k)
            if len(truth_sorted) > args.top_k
            else truth_sorted
        )

        variants = {
            "baseline": text,
            "oracle": mask_words(text, oracle_words, mask_token),
            "random": mask_words(text, random_words, mask_token),
        }
        for method in METHODS:
            variants[method] = mask_words(
                text, rankings[method][: args.top_k], mask_token
            )

        keys = list(variants.keys())
        probabilities = stego_probability([variants[k] for k in keys])
        by_key = dict(zip(keys, probabilities.tolist()))

        baseline = by_key["baseline"]
        record = {
            "dataset_index": int(index),
            "original_id": original_id,
            "num_substitutions": len(truth_words),
            "n_oracle_words": len(oracle_words),
            "n_random_words": len(random_words),
            "baseline_probability": baseline,
        }
        for key in keys:
            if key == "baseline":
                continue
            record[f"drop_{key}"] = baseline - by_key[key]

        rows.append(record)
        examined += 1

        if examined % 25 == 0:
            print(f"  {examined} samples benchmarked", flush=True)

    if examined == 0:
        raise RuntimeError(
            "No sample had all three explanation files.\n"
            "Run: python -m scripts.explain_batch --limit 200"
        )

    deletion = pd.DataFrame(rows)

    # ---------------- report ----------------
    print(f"\nSamples benchmarked: {examined}")

    print("\n" + "=" * 74)
    print("A. GROUND-TRUTH LOCALIZATION (does the ranking find the real words?)")
    print("=" * 74)
    header = (
        f"  {'method':<6}{'P@1':>8}{'P@5':>8}{'MAP':>8}"
        f"{'MRR':>8}{'rankAUC':>9}{'lift@5':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    summary_localization = {}
    for method in METHODS:
        stats = aggregate_localization(per_method_localization[method])
        summary_localization[method] = stats
        print(
            f"  {method:<6}"
            f"{stats.get('mean_precision_at_1', float('nan')):>8.3f}"
            f"{stats.get('mean_precision_at_5', float('nan')):>8.3f}"
            f"{stats.get('mean_average_precision', float('nan')):>8.3f}"
            f"{stats.get('mean_reciprocal_rank', float('nan')):>8.3f}"
            f"{stats.get('mean_ranking_auc', float('nan')):>9.3f}"
            f"{stats.get('mean_precision_at_5_lift', float('nan')):>8.2f}x"
        )

    print("\n" + "=" * 74)
    print(f"B. DELETION TEST (drop in P(stego) when top-{args.top_k} words are masked)")
    print("=" * 74)
    print(f"  {'variant':<10}{'mean drop':>12}{'std err':>10}{'% of oracle':>14}")
    print("  " + "-" * 44)
    oracle_mean = float(deletion["drop_oracle"].mean())
    for key in ["oracle"] + list(METHODS) + ["random"]:
        column = deletion[f"drop_{key}"]
        mean = float(column.mean())
        stderr = float(column.std(ddof=1) / np.sqrt(len(column)))
        share = 100.0 * mean / oracle_mean if oracle_mean else float("nan")
        label = {
            "oracle": "ORACLE",
            "random": "RANDOM",
        }.get(key, key)
        print(f"  {label:<10}{mean:>12.4f}{stderr:>10.4f}{share:>13.1f}%")
    mean_oracle_words = float(deletion["n_oracle_words"].mean())
    matched = deletion[deletion["n_oracle_words"] >= args.top_k]

    print(
        f"\n  ORACLE = up to {args.top_k} of the encoder's own substituted"
        " words (upper anchor)."
        f"\n  RANDOM = {args.top_k} unsubstituted words (lower anchor; not"
        " zero -- masking any word perturbs the input)."
        f"\n  Mean words masked by ORACLE: {mean_oracle_words:.2f}. Reviews"
        f" with fewer than {args.top_k} substitutions"
        "\n  give the oracle fewer words to mask than the methods get, so"
        " it is perturbed less."
    )

    if len(matched) >= 10:
        print(
            f"\n  Restricted to the {len(matched)} reviews with at least"
            f" {args.top_k} substitutions, where every"
            "\n  variant masks the same number of words:"
        )
        oracle_matched = float(matched["drop_oracle"].mean())
        for key in ["oracle", *METHODS, "random"]:
            value = float(matched[f"drop_{key}"].mean())
            share = (
                100.0 * value / oracle_matched
                if oracle_matched
                else float("nan")
            )
            label = {"oracle": "ORACLE", "random": "RANDOM"}.get(key, key)
            print(f"    {label:<10}{value:>12.4f}{share:>13.1f}%")

    print("\n" + "=" * 74)
    print("C. DOES THE PROXY AGREE WITH THE TRUTH?")
    print("=" * 74)
    print(f"  {'method':<6}{'Pearson r':>12}{'Spearman':>11}{'n':>7}")
    print("  " + "-" * 36)
    agreement = {}
    for method in METHODS:
        localization_scores = np.asarray(
            [
                sample.get("average_precision", np.nan)
                for sample in per_method_localization[method]
            ],
            dtype=float,
        )
        deletion_scores = deletion[f"drop_{method}"].to_numpy(dtype=float)

        n = min(len(localization_scores), len(deletion_scores))
        a = localization_scores[:n]
        b = deletion_scores[:n]
        valid = ~(np.isnan(a) | np.isnan(b))
        a, b = a[valid], b[valid]

        if a.size > 2 and a.std() > 0 and b.std() > 0:
            pearson = float(np.corrcoef(a, b)[0, 1])
            rank_a = pd.Series(a).rank().to_numpy()
            rank_b = pd.Series(b).rank().to_numpy()
            spearman = float(np.corrcoef(rank_a, rank_b)[0, 1])
        else:
            pearson = spearman = float("nan")

        agreement[method] = {
            "pearson": pearson,
            "spearman": spearman,
            "n": int(a.size),
        }
        print(f"  {method:<6}{pearson:>12.3f}{spearman:>11.3f}{a.size:>7}")

    print(
        "\n  Correlation between a sample's ground-truth localization (MAP)"
        "\n  and its deletion-test drop, for the same method on the same"
        "\n  sample. High = the proxy tracks real correctness. Low = the"
        "\n  proxy rewards something other than finding the right words."
    )

    print("\n" + "=" * 74)
    print("D. LOCALIZATION vs EMBEDDING RATE")
    print("=" * 74)
    print(f"  {'subs':>5}{'n':>6}" + "".join(f"{m + ' MAP':>11}" for m in METHODS))
    print("  " + "-" * 44)
    rate_rows = []
    for low, high in [(1, 2), (3, 3), (4, 4), (5, 5), (6, 99)]:
        entry: Dict[str, Any] = {
            "substitutions": f"{low}" if low == high else f"{low}-{high}"
        }
        count = 0
        cells = []
        for method in METHODS:
            values = [
                sample["average_precision"]
                for sample in per_method_localization[method]
                if low <= sample["num_substitutions"] <= high
                and "average_precision" in sample
            ]
            count = max(count, len(values))
            mean = float(np.mean(values)) if values else float("nan")
            entry[f"{method}_map"] = mean
            cells.append(mean)
        entry["n"] = count
        if count >= 5:
            rate_rows.append(entry)
            print(
                f"  {entry['substitutions']:>5}{count:>6}"
                + "".join(f"{c:>11.3f}" for c in cells)
            )

    report = {
        "n_samples": examined,
        "top_k": args.top_k,
        "localization": {
            m: summary_localization[m] for m in METHODS
        },
        "deletion_test": {
            key: {
                "mean_drop": float(deletion[f"drop_{key}"].mean()),
                "stderr": float(
                    deletion[f"drop_{key}"].std(ddof=1)
                    / np.sqrt(len(deletion))
                ),
            }
            for key in ["oracle", "random", *METHODS]
        },
        "proxy_truth_agreement": agreement,
        "localization_by_rate": rate_rows,
    }

    output = Path(EXPLANATION_DIR) / "explainer_benchmark.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    deletion.to_csv(
        Path(EXPLANATION_DIR) / "deletion_test_per_sample.csv", index=False
    )
    print(f"\nWritten: {output}")


if __name__ == "__main__":
    main()
