"""Every rung must fit on the real (small, synthetic) feature pipeline output and produce a finite,
uniform fraud score, so the harness in tune.py/train.py can drive them all the same way.
"""

import numpy as np
import pytest

from conftest import make_clean_frame
from fraud.features.pipeline import build_pipeline
from fraud.models.ladder import LADDER, build_estimator, fraud_score, get_step


@pytest.fixture
def train_features(features_cfg: dict):
    df = make_clean_frame(n=300, cards=8)
    pipe = build_pipeline("v1", features_cfg).fit(df)
    y = df["is_fraud"].to_numpy()
    return pipe.transform(df), y


@pytest.mark.parametrize("step", LADDER, ids=[s.key for s in LADDER])
def test_every_step_fits_and_scores(step, train_features) -> None:
    X, y = train_features
    estimator = build_estimator(step.key, {}, seed=42)
    estimator.fit(X, y)
    scores = fraud_score(estimator, X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()


def test_dummy_never_flags(train_features) -> None:
    X, y = train_features
    estimator = build_estimator("dummy", {}, seed=42)
    estimator.fit(X, y)
    assert (fraud_score(estimator, X) == 0).all()


def test_amount_rule_ranks_by_log_amt(train_features) -> None:
    X, _ = train_features
    estimator = build_estimator("amount_rule", {}, seed=42)
    estimator.fit(X, None)
    scores = fraud_score(estimator, X)
    order = np.argsort(X["log_amt"].to_numpy())
    assert np.all(np.diff(scores[order]) >= 0)


def test_get_step_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown ladder step"):
        get_step("not-a-step")


def test_tunable_steps_expose_a_valid_optuna_space() -> None:
    import optuna

    study = optuna.create_study(direction="maximize")
    for step in LADDER:
        if step.space is None:
            continue
        trial = study.ask()
        params = step.space(trial)
        estimator = step.build(params, 42)
        assert estimator is not None
