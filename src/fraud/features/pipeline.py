"""Assemble the versioned feature pipeline (v1 stateless, v2 adds card history).

Encoders, imputers and scalers live inside the pipeline, so they are fitted on whatever the caller
passes to ``fit`` (the training split) and can never drift from the model that uses them.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud.features import PIPELINE_VERSION
from fraud.features.history import (
    BEHAVIOURAL_COLUMNS,
    CardHistoryFeatures,
    velocity_column_names,
)
from fraud.features.transformers import (
    CustomerFeatures,
    GeographyFeatures,
    TransactionFeatures,
)

FEATURE_SETS = ("v1", "v2")

CATEGORICAL = ["category", "gender", "state"]
BINARY = ["is_night", "is_first_category_use", "is_first_card_txn"]
V1_NUMERIC = ["log_amt", "hour", "weekday", "age", "log_city_pop", "distance_km"]
V2_NUMERIC = [c for c in BEHAVIOURAL_COLUMNS if c not in BINARY]


def numeric_columns(feature_set: str, cfg: dict[str, Any]) -> list[str]:
    if feature_set == "v1":
        return list(V1_NUMERIC)
    return [*V1_NUMERIC, *velocity_column_names(cfg["velocity_windows_hours"]), *V2_NUMERIC]


def binary_columns(feature_set: str) -> list[str]:
    return ["is_night"] if feature_set == "v1" else list(BINARY)


def build_pipeline(feature_set: str, cfg: dict[str, Any]) -> Pipeline:
    """Unfitted feature pipeline for ``feature_set`` ('v1' or 'v2'), configured from params.yaml."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}.")
    groups: list[tuple[str, Any]] = [
        (
            "transaction",
            TransactionFeatures(cfg["night_start_hour"], cfg["night_end_hour"]),
        ),
        ("customer", CustomerFeatures()),
        ("geography", GeographyFeatures()),
    ]
    if feature_set == "v2":
        groups.append(("history", CardHistoryFeatures(cfg["velocity_windows_hours"])))

    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy=cfg["impute_strategy"])),
            ("scale", StandardScaler()),
        ]
    )
    encode = ColumnTransformer(
        [
            ("num", numeric, numeric_columns(feature_set, cfg)),
            ("bin", "passthrough", binary_columns(feature_set)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        [("features", FeatureUnion(groups, verbose_feature_names_out=False)), ("encode", encode)]
    )
    return pipeline.set_output(transform="pandas")


def transform_with_context(
    pipeline: Pipeline, X: pd.DataFrame, context: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Transform ``X`` using ``context`` (earlier stored transactions) as card history.

    Only the features of ``X`` are returned, in the order of ``X``; context rows are never scored.
    Later rows never influence earlier ones, so the context may safely be a superset of what is
    needed. A stateless (v1) pipeline ignores the context.
    """
    if context is None or context.empty:
        plain: pd.DataFrame = pipeline.transform(X)
        return plain
    combined = pd.concat([context, X], ignore_index=True)
    scored: pd.DataFrame = pipeline.transform(combined)
    return scored.iloc[len(context) :].set_axis(X.index)


def feature_list(pipeline: Pipeline, transformed: pd.DataFrame, feature_set: str) -> dict[str, Any]:
    """The pipeline's own feature list: version, set, and final column names with dtypes."""
    names = list(pipeline.get_feature_names_out())
    if names != list(transformed.columns):
        raise ValueError("Pipeline feature names do not match the transformed frame.")
    return {
        "pipeline_version": PIPELINE_VERSION,
        "feature_set": feature_set,
        "n_features": len(names),
        "features": [{"name": c, "dtype": str(transformed[c].dtype)} for c in names],
    }
