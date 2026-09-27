"""Metrics for the Phase 3 protocol (docs/plan.md).

Primary metric: recall at precision >= min_precision on validation. Secondary: PR-AUC and expected
cost (a missed fraud costs `fn_cost` times a false alarm). ROC-AUC and accuracy are not implemented
here: at 0.5% fraud they look good even for weak models and the protocol never uses them to choose.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass(frozen=True)
class ThresholdResult:
    """The operating point chosen on the precision-recall curve."""

    threshold: float
    precision: float
    recall: float
    met_budget: bool


def pr_auc(y_true: NDArray[np.int_], y_score: NDArray[np.float64]) -> float:
    return float(average_precision_score(y_true, y_score))


def recall_at_precision(
    y_true: NDArray[np.int_], y_score: NDArray[np.float64], min_precision: float
) -> ThresholdResult:
    """The highest-recall point on the PR curve with precision >= min_precision.

    Falls back to the highest-precision point available (recall 0 if nothing else reaches the
    budget) when no threshold meets it, and flags that in `met_budget`.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve appends one extra (precision=1, recall=0) point for the implicit
    # threshold of +inf ("flag nothing"); pad thresholds so all three arrays line up. Keeping this
    # point (rather than dropping it) matters: it is the correct fallback candidate whenever no
    # real threshold reaches the precision budget.
    thresholds = np.append(thresholds, np.inf)
    # The trivial last point always has precision 1.0, which trivially satisfies any budget; whether
    # the budget was genuinely met is decided by the real (non-trivial) points only.
    real_meets_budget = precision[:-1] >= min_precision
    met = bool(real_meets_budget.any())
    candidates = np.append(real_meets_budget, True)  # the trivial point is always a fallback option
    idx = int(np.argmax(np.where(candidates, recall, -1.0)))
    return ThresholdResult(float(thresholds[idx]), float(precision[idx]), float(recall[idx]), met)


def confusion_counts(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_]
) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def expected_cost(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_], fn_cost: float, fp_cost: float
) -> float:
    """Mean per-transaction cost: a missed fraud costs `fn_cost` times a false alarm (`fp_cost`)."""
    _, fp, fn, _ = confusion_counts(y_true, y_pred)
    return float(fn * fn_cost + fp * fp_cost) / len(y_true)


def flags_at_threshold(y_score: NDArray[np.float64], threshold: float) -> NDArray[np.int_]:
    return (np.asarray(y_score) >= threshold).astype(int)


def score_at_threshold(
    y_true: NDArray[np.int_],
    y_score: NDArray[np.float64],
    threshold: float,
    fn_cost: float,
    fp_cost: float,
) -> dict[str, float]:
    """All reported metrics for one fixed operating point, written to reports/*.json."""
    y_pred = flags_at_threshold(y_score, threshold)
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    # Precision is vacuously 1.0 when nothing is flagged, matching sklearn's precision_recall_curve
    # convention (recall_at_precision's fallback point) so the same threshold reports the same
    # precision here and there.
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "pr_auc": pr_auc(y_true, y_score),
        "expected_cost": expected_cost(y_true, y_pred, fn_cost, fp_cost),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def bootstrap_ci(
    y_true: NDArray[np.int_],
    y_score: NDArray[np.float64],
    threshold: float,
    fn_cost: float,
    fp_cost: float,
    *,
    n_boot: int = 1000,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs (resampling rows) for precision, recall and PR-AUC."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n = len(y_true)
    samples: dict[str, list[float]] = {"precision": [], "recall": [], "pr_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0:
            continue
        result = score_at_threshold(yt, ys, threshold, fn_cost, fp_cost)
        samples["precision"].append(result["precision"])
        samples["recall"].append(result["recall"])
        samples["pr_auc"].append(result["pr_auc"])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in samples.items()
    }
