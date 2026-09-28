"""SHAP explanations for the champion (a tree model): global summaries and per-prediction reasons.

Works on the transformed feature frame the champion's estimator actually consumes, i.e.
`full_pipeline.named_steps["model"].estimator` and `transform_with_context(...)` output, not the raw
cleaned dataframe.
"""

from typing import Any

import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray


def build_explainer(estimator: Any) -> shap.TreeExplainer:
    """A SHAP explainer for a tree-based ladder estimator (random_forest, lightgbm, xgboost)."""
    return shap.TreeExplainer(estimator)


def compute_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> NDArray[np.float64]:
    """SHAP values for the positive (fraud) class: one row per input row, one column per feature."""
    values = explainer.shap_values(X)
    if isinstance(values, list):
        # Some tree explainers (e.g. random forest) return [neg_class_values, pos_class_values].
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        # Others (e.g. some xgboost/lightgbm configurations) return (n, features, classes).
        values = values[:, :, 1]
    return values


def top_reasons(
    row_shap: NDArray[np.float64], row: pd.Series, feature_names: list[str], k: int = 5
) -> list[dict[str, Any]]:
    """The k features with the largest |SHAP value| for one prediction, most important first."""
    order = np.argsort(-np.abs(row_shap))[:k]
    return [
        {
            "feature": feature_names[i],
            "value": float(row[feature_names[i]]),
            "shap_value": float(row_shap[i]),
            "direction": "raises" if row_shap[i] > 0 else "lowers",
        }
        for i in order
    ]


def plain_language_reason(reason: dict[str, Any]) -> str:
    """One human-readable sentence for a single `top_reasons()` entry."""
    return (
        f"{reason['feature']} = {reason['value']:.3g} {reason['direction']} the fraud score "
        f"(SHAP {reason['shap_value']:+.3f})"
    )
