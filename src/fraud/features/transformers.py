"""Stateless (v1) feature transformers: transaction, customer and geography groups.

Each transformer reads the cleaned-table columns it needs and returns a DataFrame with the same
index. None of them looks at other rows, so a row scores identically alone or in a batch.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EARTH_RADIUS_KM = 6371.0088
DAYS_PER_YEAR = 365.25


def require_columns(X: pd.DataFrame, columns: Sequence[str], who: str) -> None:
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise ValueError(f"{who} needs columns {missing}, which are missing from the input.")


class FrameTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Base for transformers that need no fitting and emit a fixed list of named columns."""

    required: tuple[str, ...] = ()
    output_columns: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: object = None) -> "FrameTransformer":
        require_columns(X, self.required, type(self).__name__)
        return self

    def get_feature_names_out(self, input_features: object = None) -> npt.NDArray[np.object_]:
        return np.asarray(self.output_columns, dtype=object)

    def __sklearn_is_fitted__(self) -> bool:
        return True


class TransactionFeatures(FrameTransformer):
    """log amount, category, hour, weekday and a night flag."""

    required = ("trans_ts", "amt", "category")
    output_columns = ("log_amt", "category", "hour", "weekday", "is_night")

    def __init__(self, night_start_hour: int = 22, night_end_hour: int = 4) -> None:
        self.night_start_hour = night_start_hour
        self.night_end_hour = night_end_hour

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        require_columns(X, self.required, type(self).__name__)
        hour = X["trans_ts"].dt.hour
        # Night wraps midnight (e.g. 22:00-03:59), so it is an OR, not a range.
        night = (hour >= self.night_start_hour) | (hour < self.night_end_hour)
        return pd.DataFrame(
            {
                "log_amt": np.log1p(X["amt"].astype(float)),
                "category": X["category"].astype(object),
                "hour": hour.astype(float),
                "weekday": X["trans_ts"].dt.weekday.astype(float),
                "is_night": night.astype(float),
            },
            index=X.index,
        )


class CustomerFeatures(FrameTransformer):
    """Age at transaction, gender, log city population and state."""

    required = ("trans_ts", "dob", "gender", "city_pop", "state")
    output_columns = ("age", "gender", "log_city_pop", "state")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        require_columns(X, self.required, type(self).__name__)
        age_days = (X["trans_ts"] - X["dob"]).dt.total_seconds() / 86_400
        return pd.DataFrame(
            {
                "age": age_days / DAYS_PER_YEAR,
                "gender": X["gender"].astype(object),
                "log_city_pop": np.log1p(X["city_pop"].astype(float)),
                "state": X["state"].astype(object),
            },
            index=X.index,
        )


def haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> npt.NDArray[np.float64]:
    """Great-circle distance in km between two sets of points given in degrees."""
    p1, p2 = np.radians(lat1.to_numpy(float)), np.radians(lat2.to_numpy(float))
    dphi = p2 - p1
    dlmb = np.radians(lon2.to_numpy(float) - lon1.to_numpy(float))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return np.asarray(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)), dtype=np.float64)


class GeographyFeatures(FrameTransformer):
    """Customer-merchant distance (haversine, km)."""

    required = ("lat", "long", "merch_lat", "merch_long")
    output_columns = ("distance_km",)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        require_columns(X, self.required, type(self).__name__)
        dist = haversine_km(X["lat"], X["long"], X["merch_lat"], X["merch_long"])
        return pd.DataFrame({"distance_km": dist}, index=X.index)
