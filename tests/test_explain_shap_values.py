import numpy as np
import pytest

from conftest import make_clean_frame
from fraud.explain.shap_values import (
    build_explainer,
    compute_shap_values,
    plain_language_reason,
    top_reasons,
)
from fraud.features.pipeline import build_pipeline
from fraud.models.ladder import build_estimator


@pytest.fixture
def fitted_tree_model(features_cfg: dict):
    df = make_clean_frame(n=300, cards=8)
    pipe = build_pipeline("v1", features_cfg)
    X = pipe.fit_transform(df)
    y = df["is_fraud"].to_numpy()
    estimator = build_estimator("random_forest", {"n_estimators": 20, "max_depth": 4}, seed=42)
    estimator.fit(X, y)
    return estimator, X


def test_shap_values_shape_matches_features(fitted_tree_model) -> None:
    estimator, X = fitted_tree_model
    explainer = build_explainer(estimator)
    values = compute_shap_values(explainer, X)
    assert values.shape == (len(X), X.shape[1])
    assert np.isfinite(values).all()


def test_top_reasons_orders_by_absolute_shap(fitted_tree_model) -> None:
    estimator, X = fitted_tree_model
    explainer = build_explainer(estimator)
    values = compute_shap_values(explainer, X)
    feature_names = list(X.columns)
    reasons = top_reasons(values[0], X.iloc[0], feature_names, k=5)
    assert len(reasons) == 5
    magnitudes = [abs(r["shap_value"]) for r in reasons]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(r["direction"] in {"raises", "lowers"} for r in reasons)


def test_plain_language_reason_mentions_the_feature(fitted_tree_model) -> None:
    estimator, X = fitted_tree_model
    explainer = build_explainer(estimator)
    values = compute_shap_values(explainer, X)
    reason = top_reasons(values[0], X.iloc[0], list(X.columns), k=1)[0]
    sentence = plain_language_reason(reason)
    assert reason["feature"] in sentence
