"""Feature pipeline `features-ulb-1.0.0`: amount scaling and time-of-day only.

V1-V28 are already PCA outputs, so they pass through untouched. `Time` is seconds since the first
transaction, not a clock time: the time-of-day feature is the position in a 24 h cycle relative to
that start, encoded as sin/cos so 23:59 sits next to 00:00.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from fraud.ulb import PIPELINE_VERSION
from fraud.ulb.data import PCA_COLUMNS


class TimeOfDay(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """`Time` (seconds) -> hour_sin, hour_cos over a `seconds_per_day` cycle."""

    def __init__(self, seconds_per_day: int = 86400) -> None:
        self.seconds_per_day = seconds_per_day

    def fit(self, X: Any, y: Any = None) -> "TimeOfDay":
        return self

    def transform(self, X: Any) -> NDArray[np.float64]:
        seconds = np.asarray(X, dtype=np.float64).reshape(-1)
        angle = 2 * math.pi * (seconds % self.seconds_per_day) / self.seconds_per_day
        return np.column_stack([np.sin(angle), np.cos(angle)])

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[np.object_]:
        return np.array(["hour_sin", "hour_cos"], dtype=object)


def _log_amt_name(_: Any, names: Any) -> list[str]:
    return ["log_amt"]


def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    """Unfitted pipeline over the raw ULB columns (Time, V1-V28, Amount); scaler fits on train."""
    amount = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out=_log_amt_name)),
            ("scale", StandardScaler()),
        ]
    )
    encode = ColumnTransformer(
        [
            ("amount", amount, ["Amount"]),
            ("time", TimeOfDay(cfg["seconds_per_day"]), ["Time"]),
            ("pca", "passthrough", PCA_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("encode", encode)]).set_output(transform="pandas")


def feature_list(pipeline: Pipeline, transformed: pd.DataFrame) -> dict[str, Any]:
    names = list(pipeline.get_feature_names_out())
    if names != list(transformed.columns):
        raise ValueError("Pipeline feature names do not match the transformed frame.")
    return {
        "pipeline_version": PIPELINE_VERSION,
        "n_features": len(names),
        "features": [{"name": c, "dtype": str(transformed[c].dtype)} for c in names],
    }
