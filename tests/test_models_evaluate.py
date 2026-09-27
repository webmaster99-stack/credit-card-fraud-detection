import pandas as pd
import pytest

from conftest import make_clean_frame
from fraud.models.evaluate import evaluate_on_valid
from fraud.models.train import fit_champion_candidate


@pytest.fixture
def split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = make_clean_frame(n=500, cards=12)
    return df.iloc[:350].reset_index(drop=True), df.iloc[350:].reset_index(drop=True)


def test_evaluate_matches_the_pipelines_own_threshold_metrics(
    split: tuple[pd.DataFrame, pd.DataFrame], features_cfg: dict
) -> None:
    train, valid = split
    model_cfg = {
        "step": "logreg",
        "feature_set": "v1",
        "hyperparams": {},
        "calibration": "sigmoid",
        "min_precision": 0.5,
    }
    pipeline, train_metrics = fit_champion_candidate(train, valid, model_cfg, features_cfg, seed=42)
    metrics, curve = evaluate_on_valid(pipeline, train, valid, fn_cost=20.0, fp_cost=1.0)

    # Same threshold, same validation rows scored the same way: precision/recall must agree.
    assert metrics["precision"] == pytest.approx(train_metrics["precision"])
    assert metrics["recall"] == pytest.approx(train_metrics["recall"])
    assert metrics["threshold"] == pytest.approx(train_metrics["threshold"])
    assert set(curve) == {"precision", "recall", "thresholds"}
    assert len(curve["precision"]) == len(curve["recall"])
