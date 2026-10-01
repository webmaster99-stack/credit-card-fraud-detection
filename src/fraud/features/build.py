"""Fit both feature pipelines on the training split and check them on the validation split.

Writes the pipeline-generated feature lists and a summary. The test split is never opened here.

Usage: uv run python -m fraud.features.build
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from fraud.data.paths import PROCESSED_DIR
from fraud.features import PIPELINE_VERSION
from fraud.features.pipeline import (
    FEATURE_SETS,
    build_pipeline,
    feature_list,
    transform_with_context,
)
from fraud.params import REPO_ROOT, load_params

REPORTS_DIR = REPO_ROOT / "reports"
SUMMARY_PATH = REPORTS_DIR / "features_summary.json"


def feature_list_path(feature_set: str) -> Any:
    return REPORTS_DIR / f"feature_list_{feature_set}.json"


def build_set(
    feature_set: str, cfg: dict[str, Any], train: pd.DataFrame, valid: pd.DataFrame
) -> dict[str, Any]:
    """Fit on train, transform valid with train as card history, and validate the output."""
    pipeline = build_pipeline(feature_set, cfg)
    train_features = pipeline.fit_transform(train)
    valid_features = transform_with_context(pipeline, valid, context=train)
    for name, frame in (("train", train_features), ("valid", valid_features)):
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{feature_set}/{name}: features contain NaN or infinite values.")
    if len(valid_features) != len(valid):
        raise ValueError(f"{feature_set}: row count changed during transform.")
    listing = feature_list(pipeline, train_features, feature_set)
    feature_list_path(feature_set).write_text(
        json.dumps(listing, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return {"n_features": listing["n_features"], "train_rows": len(train_features)}


def main() -> None:
    params = load_params()
    cfg = params["features"]
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"pipeline_version": PIPELINE_VERSION}
    for feature_set in FEATURE_SETS:
        summary[feature_set] = build_set(feature_set, cfg, train, valid)
        print(f"{feature_set}: {summary[feature_set]['n_features']} features")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
