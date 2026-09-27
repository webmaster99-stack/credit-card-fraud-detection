import numpy as np
import pytest

from conftest import make_clean_frame
from fraud.features.pipeline import build_pipeline
from fraud.models.estimator import CalibratedThresholdClassifier
from fraud.models.ladder import build_estimator


@pytest.fixture
def split_features(features_cfg: dict):
    df = make_clean_frame(n=400, cards=10)
    train, valid = df.iloc[:250].reset_index(drop=True), df.iloc[250:].reset_index(drop=True)
    pipe = build_pipeline("v1", features_cfg)
    X_train = pipe.fit_transform(train)
    X_valid = pipe.transform(valid)
    return X_train, train["is_fraud"].to_numpy(), X_valid, valid["is_fraud"].to_numpy()


def test_fit_then_calibrate_then_threshold(split_features) -> None:
    X_train, y_train, X_valid, y_valid = split_features
    model = CalibratedThresholdClassifier(build_estimator("logreg", {}, seed=42), "sigmoid")
    model.fit(X_train, y_train)
    proba_before = model.predict_proba(X_valid)
    assert proba_before.shape == (len(X_valid), 2)
    np.testing.assert_allclose(proba_before.sum(axis=1), 1.0)

    model.calibrate(X_valid, y_valid)
    model.set_threshold(0.7)
    assert model.threshold_ == 0.7
    preds = model.predict(X_valid)
    assert set(np.unique(preds)) <= {0, 1}
    proba_after = model.predict_proba(X_valid)[:, 1]
    np.testing.assert_array_equal(preds, (proba_after >= 0.7).astype(int))


def test_uncalibrated_model_defaults_to_half_threshold(split_features) -> None:
    X_train, y_train, X_valid, _ = split_features
    model = CalibratedThresholdClassifier(build_estimator("logreg", {}, seed=42), "none")
    model.fit(X_train, y_train)
    assert model.threshold_ == 0.5
    preds = model.predict(X_valid)
    proba = model.predict_proba(X_valid)[:, 1]
    np.testing.assert_array_equal(preds, (proba >= 0.5).astype(int))
