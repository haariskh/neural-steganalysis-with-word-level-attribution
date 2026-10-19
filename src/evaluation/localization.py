"""
Scoring word-level attribution against the substitution ground truth.

The synonym encoder records every word it replaced. That record is the
only ground truth available for the question requirement 3 actually
asks: when the detector says "stego", are the words it points at the
words that were tampered with?

Producing attributions is not the same as producing *correct*
attributions. Nothing in an attribution map is self-validating, so this
module scores a ranked word list from any explainer (IG, LIME, SHAP)
against the recorded substitutions, and -- importantly -- against the
random-ranking baseline for the same review.

Why the baseline matters
------------------------
Precision@10 of 0.30 sounds strong and is meaningless on its own. A
review with 200 unique words and 8 substitutions yields an expected
precision@10 of 8/200 = 0.04 from random ranking alone. The informative
quantity is the lift over that baseline, which is what
`localization_metrics` reports as `precision_at_k_lift`.

Known limitation (state this in the write-up)
---------------------------------------------
Matching is by word identity, not token position. Explainer outputs are
sequences of aggregated word pieces without reliable indices back into
the original tokenisation, so a replacement word that also occurs
elsewhere in the review unmodified will be counted as a hit wherever it
is ranked. This inflates scores slightly. The effect is bounded by
`ambiguous_truth_words` in the per-sample output, which counts
ground-truth words that occur more than once in the text; report it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


# --------------------------------------------------------------
# Word normalisation
# --------------------------------------------------------------

def normalize_word(value: Any) -> str:
    """
    Normalise a token for comparison across explainers.

    Lowercases, strips the BERT word-piece marker, and trims leading and
    trailing non-alphanumerics so that "Wonderful," and "wonderful"
    compare equal.
    """

    text = str(value).strip().lower()

    if text.startswith("##"):
        text = text[2:]

    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)

    return text


def _is_scorable(word: str) -> bool:
    return len(word) >= 2 and bool(re.search(r"[a-z0-9]", word))


# --------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------

def load_substitution_ground_truth(
    metadata_path: str | Path,
) -> Dict[int, Dict[str, Any]]:
    """
    Load the encoder's substitution record, keyed by original_id.

    Requires the `original_id` field written by
    scripts/generate_synonym_stega.py. Metadata produced before that
    field was added cannot be joined to the shuffled detector dataset
    and is rejected here rather than silently mis-aligned.
    """

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    records = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a list of metadata records in {metadata_path}."
        )

    ground_truth: Dict[int, Dict[str, Any]] = {}
    missing_id = 0

    for record in records:
        if "original_id" not in record:
            missing_id += 1
            continue

        substitutions = record.get("substitutions", [])

        replaced_words = {
            normalize_word(item.get("replacement_word", ""))
            for item in substitutions
        }
        replaced_words = {w for w in replaced_words if _is_scorable(w)}

        original_words = {
            normalize_word(item.get("original_word", ""))
            for item in substitutions
        }
        original_words = {w for w in original_words if _is_scorable(w)}

        ground_truth[int(record["original_id"])] = {
            "replacement_words": replaced_words,
            "original_words": original_words,
            "num_substitutions": int(record.get("num_substitutions", 0)),
            "stego_text": record.get("stego_text", ""),
        }

    if missing_id:
        raise ValueError(
            f"{missing_id} of {len(records)} metadata records have no "
            "'original_id'. Regenerate the stego split with the current "
            "scripts/generate_synonym_stega.py so attributions can be "
            "joined to ground truth."
        )

    return ground_truth


# --------------------------------------------------------------
# Explainer output -> ranked word list
# --------------------------------------------------------------

def ranked_words_from_scores(
    items: Iterable[Dict[str, Any]],
    word_key: str,
    score_key: str,
    use_absolute: bool = True,
) -> List[str]:
    """
    Turn scored tokens into a deduplicated word ranking, strongest first.

    `use_absolute=True` ranks by |score|, treating evidence for either
    class as equally "important". Set it False to rank only by evidence
    pushing toward the explained class.
    """

    scored: List[tuple] = []

    for item in items:
        word = normalize_word(item.get(word_key, ""))

        if not _is_scorable(word):
            continue

        try:
            score = float(item.get(score_key, 0.0))
        except (TypeError, ValueError):
            continue

        scored.append((abs(score) if use_absolute else score, word))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    ranked: List[str] = []
    seen: Set[str] = set()

    for _, word in scored:
        if word not in seen:
            ranked.append(word)
            seen.add(word)

    return ranked


def ranked_words_from_ig(explanation: Dict[str, Any]) -> List[str]:
    """Ranked words from an IntegratedGradientsExplainer JSON."""
    return ranked_words_from_scores(
        explanation["words"], "word", "attribution"
    )


def ranked_words_from_lime(
    explanation: Dict[str, Any],
    predicted_label: int,
) -> List[str]:
    """Ranked words from the LIME JSON written by scripts/explain_all.py."""

    class_explanations = explanation["class_explanations"]

    items = class_explanations.get(
        str(predicted_label),
        class_explanations.get(predicted_label, []),
    )

    return ranked_words_from_scores(items, "word", "weight")


def ranked_words_from_shap(
    explanation: Dict[str, Any],
    predicted_label: int,
) -> List[str]:
    """Ranked words from the SHAP JSON written by scripts/explain_all.py."""

    tokens = explanation["tokens"]
    scores = explanation.get("target_attributions")

    if scores is None:
        values = np.asarray(explanation["values"])
        if values.ndim == 3:
            values = values[0]
        scores = values[:, predicted_label].tolist()

    if len(tokens) != len(scores):
        raise ValueError(
            f"SHAP token/score length mismatch: "
            f"{len(tokens)} vs {len(scores)}"
        )

    items = [
        {"word": token, "score": score}
        for token, score in zip(tokens, scores)
    ]

    return ranked_words_from_scores(items, "word", "score")


# --------------------------------------------------------------
# Localization metrics
# --------------------------------------------------------------

def localization_metrics(
    ranked_words: Sequence[str],
    truth_words: Set[str],
    text: Optional[str] = None,
    k_values: Sequence[int] = (1, 5, 10, 20),
) -> Dict[str, Any]:
    """
    Score one ranked word list against one review's substituted words.

    Args:
        ranked_words: Explainer output, most important word first.
        truth_words: Normalised replacement words for this review.
        text: The review text. Used to size the random baseline and to
            count ambiguous ground-truth words. Optional but recommended.
        k_values: Cut-offs for precision@k / recall@k.

    Returns:
        Per-sample metrics. `precision_at_k_lift` is precision@k divided
        by the expected precision of a random ranking; 1.0 means the
        explainer is no better than chance.
    """

    ranked = [w for w in ranked_words if _is_scorable(w)]
    truth = {w for w in truth_words if _is_scorable(w)}

    result: Dict[str, Any] = {
        "num_truth_words": len(truth),
        "num_ranked_words": len(ranked),
    }

    if not truth:
        result["skipped_reason"] = "no ground-truth substitutions"
        return result

    if not ranked:
        result["skipped_reason"] = "explainer produced no scorable words"
        return result

    # Candidate pool size drives the random baseline.
    if text is not None:
        pool = {
            normalize_word(token)
            for token in re.findall(r"\w+", str(text))
        }
        pool = {w for w in pool if _is_scorable(w)}
        pool_size = max(len(pool), len(ranked))

        ambiguous = 0
        lowered = str(text).lower()
        for word in truth:
            if len(re.findall(rf"(?<!\w){re.escape(word)}(?!\w)", lowered)) > 1:
                ambiguous += 1
        result["ambiguous_truth_words"] = ambiguous
    else:
        pool_size = len(ranked)

    result["candidate_pool_size"] = int(pool_size)

    # Chance precision: fraction of the pool that is ground truth.
    chance_precision = len(truth) / pool_size if pool_size else 0.0
    result["chance_precision"] = float(chance_precision)

    hits = [1 if word in truth else 0 for word in ranked]

    for k in k_values:
        top_k = hits[:k]
        n_hits = int(sum(top_k))

        precision_at_k = n_hits / k if k > 0 else 0.0
        recall_at_k = n_hits / len(truth)

        result[f"precision_at_{k}"] = float(precision_at_k)
        result[f"recall_at_{k}"] = float(recall_at_k)
        result[f"hit_at_{k}"] = int(n_hits > 0)
        result[f"precision_at_{k}_lift"] = (
            float(precision_at_k / chance_precision)
            if chance_precision > 0
            else float("nan")
        )

    # Average precision over the full ranking.
    running_hits = 0
    precision_sum = 0.0

    for position, hit in enumerate(hits, start=1):
        if hit:
            running_hits += 1
            precision_sum += running_hits / position

    result["average_precision"] = float(
        precision_sum / len(truth) if truth else 0.0
    )

    # Rank of the first correctly identified substituted word.
    first_hit = next(
        (i for i, hit in enumerate(hits, start=1) if hit), None
    )
    result["reciprocal_rank"] = float(1.0 / first_hit) if first_hit else 0.0
    result["first_hit_rank"] = int(first_hit) if first_hit else -1

    # Rank-based AUC: probability a truth word outranks a non-truth word.
    n_truth_ranked = sum(hits)
    n_other = len(hits) - n_truth_ranked

    if n_truth_ranked > 0 and n_other > 0:
        # Ranks are positions in `ranked`; lower is better.
        truth_ranks = [i for i, hit in enumerate(hits) if hit]
        concordant = sum(
            (len(hits) - 1 - rank) - (n_truth_ranked - 1 - order)
            for order, rank in enumerate(truth_ranks)
        )
        result["ranking_auc"] = float(
            concordant / (n_truth_ranked * n_other)
        )
    else:
        result["ranking_auc"] = float("nan")

    return result


def aggregate_localization(
    per_sample: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Mean and standard error of per-sample localization metrics.

    Samples carrying a `skipped_reason` are excluded from the means and
    counted separately, so a high score cannot come from quietly
    dropping the hard cases.
    """

    scored = [row for row in per_sample if "skipped_reason" not in row]
    skipped = [row for row in per_sample if "skipped_reason" in row]

    summary: Dict[str, Any] = {
        "n_scored": len(scored),
        "n_skipped": len(skipped),
    }

    if skipped:
        reasons: Dict[str, int] = {}
        for row in skipped:
            reasons[row["skipped_reason"]] = (
                reasons.get(row["skipped_reason"], 0) + 1
            )
        summary["skipped_reasons"] = reasons

    if not scored:
        return summary

    numeric_keys = {
        key
        for row in scored
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }

    for key in sorted(numeric_keys):
        values = np.asarray(
            [
                float(row[key])
                for row in scored
                if key in row and not np.isnan(float(row[key]))
            ],
            dtype=float,
        )

        if values.size == 0:
            continue

        summary[f"mean_{key}"] = float(values.mean())
        summary[f"stderr_{key}"] = (
            float(values.std(ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else 0.0
        )

    return summary
