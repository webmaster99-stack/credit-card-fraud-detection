import pandas as pd
import pytest

from conftest import make_clean_frame
from fraud.models.evaluate_test import evaluate_on_test
from fraud.models.train import fit_champion_candidate


@pytest.fixture
def split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = make_clean_frame(n=600, cards=15)
    train = df.iloc[:350].reset_index(drop=True)
    valid = df.iloc[350:480].reset_index(drop=True)
    test = df.iloc[480:].reset_index(drop=True)
    return train, valid, test


def test_evaluate_on_test_uses_the_fixed_threshold_and_reports_ci(
    split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], features_cfg: dict
) -> None:
    train, valid, test = split
    model_cfg = {
        "step": "logreg",
        "feature_set": "v1",
        "hyperparams": {},
        "calibration": "sigmoid",
        "min_precision": 0.5,
    }
    pipeline, train_metrics = fit_champion_candidate(train, valid, model_cfg, features_cfg, seed=42)
    threshold_before = pipeline.named_steps["model"].threshold_

    result = evaluate_on_test(
        pipeline, train, valid, test, fn_cost=20.0, fp_cost=1.0, n_boot=50, seed=42
    )

    # The threshold is never re-picked on test.
    assert pipeline.named_steps["model"].threshold_ == threshold_before
    assert result["threshold"] == pytest.approx(train_metrics["threshold"])
    assert "confidence_intervals" in result
    assert set(result["confidence_intervals"]) == {"precision", "recall", "pr_auc"}
    for lo, hi in result["confidence_intervals"].values():
        assert lo <= hi


def test_evaluate_on_test_uses_train_and_valid_as_history_context(
    split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], features_cfg: dict
) -> None:
    train, valid, test = split
    model_cfg = {
        "step": "logreg",
        "feature_set": "v2",
        "hyperparams": {},
        "calibration": "sigmoid",
        "min_precision": 0.5,
    }
    pipeline, _ = fit_champion_candidate(train, valid, model_cfg, features_cfg, seed=42)
    result = evaluate_on_test(
        pipeline, train, valid, test, fn_cost=20.0, fp_cost=1.0, n_boot=20, seed=42
    )
    assert 0.0 <= result["recall"] <= 1.0
