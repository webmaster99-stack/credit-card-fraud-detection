import pandas as pd
import pytest

from conftest import make_clean_frame
from fraud.models.error_analysis import group_metrics, sample_errors, score_test
from fraud.models.train import fit_champion_candidate


def test_group_metrics_computes_recall_and_false_alarm_rate() -> None:
    y_true = pd.Series([1, 1, 0, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 1, 1, 0])
    group = pd.Series(["A", "A", "A", "B", "B", "B"])
    result = group_metrics(y_true, y_pred, group)
    assert set(result) == {"A", "B"}
    # Group A: frauds at idx 0,1 (pred 1,0) -> recall 0.5; legit at idx 2 (pred 0) -> FA rate 0.0
    assert result["A"]["recall"] == pytest.approx(0.5)
    assert result["A"]["false_alarm_rate"] == pytest.approx(0.0)
    # Group B: fraud at idx 4 (pred 1) -> recall 1.0; legit at idx 3,5 (pred 1,0) -> FA rate 0.5
    assert result["B"]["recall"] == pytest.approx(1.0)
    assert result["B"]["false_alarm_rate"] == pytest.approx(0.5)


def test_group_metrics_none_when_group_has_no_fraud_or_no_legit() -> None:
    y_true = pd.Series([1, 1])
    y_pred = pd.Series([1, 0])
    group = pd.Series(["A", "A"])
    result = group_metrics(y_true, y_pred, group)
    assert result["A"]["false_alarm_rate"] is None
    assert result["A"]["recall"] == pytest.approx(0.5)


def test_sample_errors_caps_at_k_and_labels_correctly() -> None:
    test = pd.DataFrame(
        {
            "is_fraud": [1, 1, 1, 0, 0, 0],
            "predicted": [0, 0, 1, 1, 1, 0],
            "trans_ts": pd.date_range("2020-10-01", periods=6, freq="h"),
            "category": "grocery_pos",
            "amt": 10.0,
            "gender": "F",
            "state": "IL",
            "city_pop": 1000,
            "fraud_score": 0.5,
        }
    )
    result = sample_errors(test, seed=0, k=1)
    assert result["n_missed_frauds"] == 2  # rows 0, 1
    assert result["n_false_alarms"] == 2  # rows 3, 4
    assert len(result["missed_fraud_sample"]) == 1
    assert len(result["false_alarm_sample"]) == 1


def test_score_test_returns_series_aligned_to_test_index(features_cfg: dict) -> None:
    df = make_clean_frame(n=400, cards=10)
    train, valid, test = (
        df.iloc[:200].reset_index(drop=True),
        df.iloc[200:300].reset_index(drop=True),
        df.iloc[300:].reset_index(drop=True),
    )
    model_cfg = {
        "step": "logreg",
        "feature_set": "v1",
        "hyperparams": {},
        "calibration": "sigmoid",
        "min_precision": 0.5,
    }
    pipeline, _ = fit_champion_candidate(train, valid, model_cfg, features_cfg, seed=42)
    score, threshold = score_test(pipeline, train, valid, test)
    assert list(score.index) == list(test.index)
    assert isinstance(threshold, float)
    assert ((score >= 0) & (score <= 1)).all()
