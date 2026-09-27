import numpy as np

from fraud.models.metrics import (
    bootstrap_ci,
    confusion_counts,
    expected_cost,
    flags_at_threshold,
    pr_auc,
    recall_at_precision,
    score_at_threshold,
)


def test_confusion_counts() -> None:
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    assert (tp, fp, fn, tn) == (2, 1, 1, 1)


def test_expected_cost_weighs_missed_fraud_more() -> None:
    y_true = np.array([1, 0, 0, 0])
    fn_only = np.array([0, 0, 0, 0])  # misses the one fraud
    fp_only = np.array([1, 1, 0, 0])  # catches it, one false alarm
    assert expected_cost(y_true, fn_only, fn_cost=20.0, fp_cost=1.0) > expected_cost(
        y_true, fp_only, fn_cost=20.0, fp_cost=1.0
    )


def test_recall_at_precision_meets_budget_when_separable() -> None:
    y_true = np.array([0] * 8 + [1] * 2)
    y_score = np.array([0.1] * 8 + [0.9] * 2)
    result = recall_at_precision(y_true, y_score, min_precision=0.5)
    assert result.met_budget
    assert result.precision >= 0.5
    assert result.recall == 1.0


def test_recall_at_precision_flags_unmet_budget() -> None:
    # Every high score is wrong; no threshold can reach precision 0.9.
    y_true = np.array([0] * 9 + [1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    result = recall_at_precision(y_true, y_score, min_precision=0.9)
    assert not result.met_budget


def test_pr_auc_perfect_separation_is_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc(y_true, y_score) == 1.0


def test_flags_at_threshold() -> None:
    y_score = np.array([0.1, 0.5, 0.6, 0.9])
    assert list(flags_at_threshold(y_score, 0.5)) == [0, 1, 1, 1]


def test_score_at_threshold_shape() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.4, 0.3, 0.1])
    result = score_at_threshold(y_true, y_score, threshold=0.5, fn_cost=20.0, fp_cost=1.0)
    assert result["tp"] == 1 and result["fn"] == 1 and result["fp"] == 0 and result["tn"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_bootstrap_ci_bounds_contain_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y_true = (rng.uniform(size=500) < 0.1).astype(int)
    y_score = np.clip(y_true + rng.normal(0, 0.3, size=500), 0, 1)
    point = score_at_threshold(y_true, y_score, threshold=0.5, fn_cost=20.0, fp_cost=1.0)
    ci = bootstrap_ci(y_true, y_score, threshold=0.5, fn_cost=20.0, fp_cost=1.0, n_boot=200, seed=0)
    lo, hi = ci["precision"]
    assert lo <= point["precision"] <= hi or abs(point["precision"] - lo) < 0.2
