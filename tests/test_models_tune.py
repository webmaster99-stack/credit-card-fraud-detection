from pathlib import Path

import mlflow
import numpy as np
import pytest

from conftest import make_clean_frame
from fraud.features.pipeline import build_pipeline
from fraud.models.results import load_ladder_results
from fraud.models.tune import tune_step


@pytest.fixture(autouse=True)
def local_mlflow_tracking(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test-tune")


@pytest.fixture
def split_features(features_cfg: dict):
    df = make_clean_frame(n=300, cards=8)
    train, valid = df.iloc[:200].reset_index(drop=True), df.iloc[200:].reset_index(drop=True)
    pipe = build_pipeline("v1", features_cfg)
    X_train = pipe.fit_transform(train)
    X_valid = pipe.transform(valid)
    return X_train, train["is_fraud"].to_numpy(), X_valid, valid["is_fraud"].to_numpy()


def test_tune_step_records_a_result(tmp_path: Path, split_features) -> None:
    X_train, y_train, X_valid, y_valid = split_features
    results_path = tmp_path / "model_ladder.json"
    result = tune_step(
        "logreg",
        X_train,
        y_train,
        X_valid,
        y_valid,
        min_precision=0.5,
        n_trials=3,
        seed=42,
        feature_set="v1",
        results_path=results_path,
    )
    assert result.key == "logreg"
    assert result.n_trials == 3
    assert 0.0 <= result.recall_at_precision <= 1.0
    saved = load_ladder_results(results_path)
    assert len(saved) == 1
    assert saved[0]["key"] == "logreg"


def test_non_tunable_step_runs_a_single_trial(tmp_path: Path, split_features) -> None:
    # A realistic (rare-event) fraud rate: "flag everyone" (the dummy's only real operating point)
    # falls well short of a 50% precision budget, so the fallback "flag nothing" point (recall 0)
    # wins, matching the ladder's floor.
    X_train, y_train, X_valid, _ = split_features
    y_valid_rare = np.zeros(len(X_valid), dtype=int)
    y_valid_rare[:2] = 1
    result = tune_step(
        "dummy",
        X_train,
        y_train,
        X_valid,
        y_valid_rare,
        min_precision=0.5,
        n_trials=10,
        seed=42,
        feature_set="v1",
        results_path=tmp_path / "model_ladder.json",
    )
    assert result.n_trials == 1
    assert result.recall_at_precision == 0.0
    assert not result.met_budget


def test_nested_true_runs_as_a_child_of_an_active_run(tmp_path: Path, split_features) -> None:
    X_train, y_train, X_valid, y_valid = split_features
    with mlflow.start_run(run_name="sweep-parent"):
        result = tune_step(
            "logreg",
            X_train,
            y_train,
            X_valid,
            y_valid,
            min_precision=0.5,
            n_trials=2,
            seed=42,
            feature_set="v1",
            results_path=tmp_path / "model_ladder.json",
            nested=True,
        )
    assert result.key == "logreg"


def test_rerunning_a_step_overwrites_its_row(tmp_path: Path, split_features) -> None:
    X_train, y_train, X_valid, y_valid = split_features
    results_path = tmp_path / "model_ladder.json"
    for _ in range(2):
        tune_step(
            "logreg",
            X_train,
            y_train,
            X_valid,
            y_valid,
            min_precision=0.5,
            n_trials=2,
            seed=42,
            feature_set="v1",
            results_path=results_path,
        )
    assert len(load_ladder_results(results_path)) == 1
