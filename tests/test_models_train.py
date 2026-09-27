import numpy as np
import pandas as pd
import pytest

from conftest import make_clean_frame
from fraud.models.train import fit_champion_candidate


@pytest.fixture
def split(clean_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = make_clean_frame(n=500, cards=12)
    return df.iloc[:350].reset_index(drop=True), df.iloc[350:].reset_index(drop=True)


@pytest.mark.parametrize("step", ["dummy", "logreg"])
def test_fit_champion_candidate_v1(
    step: str, split: tuple[pd.DataFrame, pd.DataFrame], features_cfg: dict
) -> None:
    train, valid = split
    model_cfg = {
        "step": step,
        "feature_set": "v1",
        "hyperparams": {},
        "calibration": "sigmoid",
        "min_precision": 0.5,
    }
    pipeline, metrics = fit_champion_candidate(train, valid, model_cfg, features_cfg, seed=42)

    assert metrics["step"] == step
    assert metrics["feature_set"] == "v1"
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert isinstance(metrics["met_budget"], bool)

    proba = pipeline.predict_proba(valid)
    assert proba.shape == (len(valid), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    preds = pipeline.predict(valid)
    assert set(np.unique(preds)) <= {0, 1}


def test_fit_champion_candidate_v2_uses_history(
    split: tuple[pd.DataFrame, pd.DataFrame], features_cfg: dict
) -> None:
    train, valid = split
    model_cfg = {
        "step": "logreg",
        "feature_set": "v2",
        "hyperparams": {},
        "calibration": "isotonic",
        "min_precision": 0.5,
    }
    pipeline, metrics = fit_champion_candidate(train, valid, model_cfg, features_cfg, seed=42)
    assert metrics["feature_set"] == "v2"
    assert pipeline.named_steps["model"].threshold_ == metrics["threshold"]
