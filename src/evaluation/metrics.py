"""
Shared classification metrics for the steganalysis detector.

The detector is evaluated in two regimes:

1. At the fixed 0.5 decision threshold, which is what argmax over the
   softmax gives you. This is the number most papers quote.

2. At a threshold selected on the *validation* split and then applied
   unchanged to the test split. Selecting a threshold on the test split
   and reporting the result on that same split leaks the test labels and
   inflates the figure, so the two-split protocol here is the honest one.

Threshold-free separability (ROC-AUC) is reported alongside both, because
accuracy at any single threshold is upper-bounded by the separability of
the score distributions. If accuracy is already close to that bound,
moving the threshold cannot help and only a stronger signal will.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


THRESHOLD_CRITERIA = ("balanced_accuracy", "f1", "youden", "accuracy")


def compute_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute the full metric suite at one decision threshold.

    Args:
        y_true: Ground-truth labels (0 = cover, 1 = stego).
        y_score: P(stego) for each sample.
        threshold: Decision threshold applied to y_score.

    Returns:
        Metric name -> value. Threshold-free metrics (ROC-AUC, average
        precision) do not depend on `threshold`.
    """

    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)

    if y_true_array.shape != y_score_array.shape:
        raise ValueError(
            f"y_true and y_score length differ: "
            f"{y_true_array.shape} vs {y_score_array.shape}"
        )

    if y_true_array.size == 0:
        raise ValueError("Cannot compute metrics on an empty set.")

    invalid = set(np.unique(y_true_array)) - {0, 1}
    if invalid:
        raise ValueError(f"y_true must be binary, found labels: {invalid}")

    y_prediction = (y_score_array >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_array,
        y_prediction,
        average="binary",
        zero_division=0,
    )

    # ROC-AUC and AP are undefined when only one class is present.
    both_classes_present = len(np.unique(y_true_array)) == 2

    roc_auc = (
        float(roc_auc_score(y_true_array, y_score_array))
        if both_classes_present
        else float("nan")
    )

    average_precision = (
        float(average_precision_score(y_true_array, y_score_array))
        if both_classes_present
        else float("nan")
    )

    matrix = confusion_matrix(
        y_true_array,
        y_prediction,
        labels=[0, 1],
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    specificity = (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, y_prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, y_prediction)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "n_samples": int(y_true_array.size),
        "n_positive": int(y_true_array.sum()),
    }


def select_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    criterion: str = "balanced_accuracy",
    n_grid: int = 512,
) -> Tuple[float, float]:
    """
    Pick the decision threshold maximising `criterion`.

    Call this on the VALIDATION split only, then pass the returned
    threshold to compute_metrics on the test split.

    Implemented as a single vectorised sweep: scores are sorted once and
    the confusion matrix at every candidate threshold is derived from
    cumulative sums. A naive loop calling a metric function per candidate
    is O(candidates x samples) and takes minutes on a split of this size.

    `n_grid` is accepted for backward compatibility and ignored: sweeping
    the observed scores evaluates every threshold that can change a
    prediction, so a uniform grid adds nothing.

    Returns:
        (threshold, criterion_value_at_that_threshold)
    """

    if criterion not in THRESHOLD_CRITERIA:
        raise ValueError(
            f"criterion must be one of {THRESHOLD_CRITERIA}, got {criterion!r}"
        )

    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)

    if y_true_array.shape != y_score_array.shape:
        raise ValueError("y_true and y_score must have the same length.")

    if y_true_array.size == 0:
        raise ValueError("Cannot select a threshold on an empty set.")

    n_positive = int(y_true_array.sum())
    n_negative = int(y_true_array.size - n_positive)

    # Descending score order: at a threshold just at scores[i], exactly
    # the first i+1 samples are predicted positive.
    order = np.argsort(-y_score_array, kind="mergesort")
    sorted_scores = y_score_array[order]
    sorted_labels = y_true_array[order]

    true_positive = np.cumsum(sorted_labels)
    predicted_positive = np.arange(1, y_true_array.size + 1)
    false_positive = predicted_positive - true_positive

    # Prepend the "predict nothing positive" operating point.
    true_positive = np.concatenate([[0], true_positive])
    false_positive = np.concatenate([[0], false_positive])

    false_negative = n_positive - true_positive
    true_negative = n_negative - false_positive

    with np.errstate(divide="ignore", invalid="ignore"):
        if criterion == "accuracy":
            value = (true_positive + true_negative) / y_true_array.size

        elif criterion == "balanced_accuracy":
            sensitivity = np.divide(
                true_positive, n_positive,
                out=np.zeros_like(true_positive, dtype=float),
                where=n_positive > 0,
            )
            specificity = np.divide(
                true_negative, n_negative,
                out=np.zeros_like(true_negative, dtype=float),
                where=n_negative > 0,
            )
            value = (sensitivity + specificity) / 2.0

        elif criterion == "f1":
            denominator = 2 * true_positive + false_positive + false_negative
            value = np.divide(
                2 * true_positive, denominator,
                out=np.zeros(denominator.shape, dtype=float),
                where=denominator > 0,
            )

        else:  # youden
            sensitivity = np.divide(
                true_positive, n_positive,
                out=np.zeros_like(true_positive, dtype=float),
                where=n_positive > 0,
            )
            specificity = np.divide(
                true_negative, n_negative,
                out=np.zeros_like(true_negative, dtype=float),
                where=n_negative > 0,
            )
            value = sensitivity + specificity - 1.0

    best_index = int(np.argmax(value))

    # Index 0 is the empty-positive point: a threshold above every score.
    if best_index == 0:
        threshold = float(np.nextafter(sorted_scores[0], np.inf))
    else:
        threshold = float(sorted_scores[best_index - 1])

    return threshold, float(value[best_index])


def bootstrap_confidence_interval(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metric: str = "accuracy",
    threshold: float = 0.5,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Percentile bootstrap confidence interval for one metric.

    A single test figure with no interval says nothing about how much of
    the difference between two runs is noise. Report the interval.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")

    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)

    n_samples = y_true_array.size
    rng = np.random.default_rng(seed)

    values = []

    for _ in range(n_resamples):
        index = rng.integers(0, n_samples, n_samples)

        resampled_true = y_true_array[index]

        # Skip degenerate resamples for metrics that need both classes.
        if len(np.unique(resampled_true)) < 2:
            continue

        resampled_metrics = compute_metrics(
            resampled_true,
            y_score_array[index],
            threshold=threshold,
        )
        values.append(resampled_metrics[metric])

    if not values:
        raise ValueError(
            "Every bootstrap resample was single-class; "
            "the sample is too small or too imbalanced."
        )

    values_array = np.asarray(values, dtype=float)
    alpha = (1.0 - confidence) / 2.0

    point = compute_metrics(
        y_true_array, y_score_array, threshold=threshold
    )[metric]

    return {
        "metric": metric,
        "point_estimate": float(point),
        "lower": float(np.quantile(values_array, alpha)),
        "upper": float(np.quantile(values_array, 1.0 - alpha)),
        "confidence": float(confidence),
        "n_effective_resamples": int(values_array.size),
    }


def accuracy_ceiling_from_auc(roc_auc: float) -> Dict[str, float]:
    """
    Accuracy attainable at the best possible threshold, estimated from
    ROC-AUC under an equal-variance binormal model.

    This is a MODEL-BASED estimate, not a guarantee: it assumes the two
    score distributions are Gaussian with equal variance. Real score
    distributions deviate from that, so treat the number as a guide to
    whether threshold tuning is worth attempting at all.

    Interpretation: if observed accuracy is already at or above this
    estimate, the decision threshold is not what is holding the detector
    back -- the separability of the scores is. Only a stronger signal
    (or a better model) raises it.
    """

    from statistics import NormalDist

    if not 0.0 < roc_auc < 1.0:
        raise ValueError("roc_auc must lie strictly between 0 and 1.")

    normal = NormalDist()

    # AUC = Phi(d' / sqrt(2))  =>  d' = sqrt(2) * Phi^-1(AUC)
    separability = (2.0 ** 0.5) * normal.inv_cdf(roc_auc)

    return {
        "roc_auc": float(roc_auc),
        "separability_d_prime": float(separability),
        "estimated_max_accuracy": float(normal.cdf(separability / 2.0)),
    }


def auc_required_for_accuracy(target_accuracy: float) -> float:
    """
    ROC-AUC needed to make `target_accuracy` reachable at the best
    threshold, under the same equal-variance binormal model.

    Useful for sanity-checking an accuracy target before chasing it.
    """

    from statistics import NormalDist

    if not 0.0 < target_accuracy < 1.0:
        raise ValueError("target_accuracy must lie strictly between 0 and 1.")

    normal = NormalDist()

    separability = 2.0 * normal.inv_cdf(target_accuracy)

    return float(normal.cdf(separability / (2.0 ** 0.5)))
