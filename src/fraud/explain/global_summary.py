"""Global SHAP summary for the champion: mean |SHAP value| per feature on a validation sample.

Read-only, like error_analysis.py: explains the already-trained champion, does not touch it.

Usage: uv run python -m fraud.explain.global_summary
"""

import json

import joblib
import numpy as np
import pandas as pd

from fraud.data.paths import PROCESSED_DIR
from fraud.explain.shap_values import build_explainer, compute_shap_values
from fraud.features.pipeline import transform_with_context
from fraud.models.train import MODEL_PATH
from fraud.params import REPO_ROOT, load_params

REPORTS_DIR = REPO_ROOT / "reports"
SHAP_SUMMARY_PATH = REPORTS_DIR / "shap_summary.json"
SAMPLE_SIZE = 5000


def main() -> None:
    params = load_params()
    pipeline = joblib.load(MODEL_PATH)
    feature_pipeline = pipeline.named_steps["features"]
    estimator = pipeline.named_steps["model"].estimator

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")
    X_valid = transform_with_context(feature_pipeline, valid, context=train)

    sample = X_valid.sample(n=min(SAMPLE_SIZE, len(X_valid)), random_state=params["seed"])
    explainer = build_explainer(estimator)
    values = compute_shap_values(explainer, sample)

    mean_abs = np.abs(values).mean(axis=0)
    ranking = sorted(zip(sample.columns, mean_abs.tolist(), strict=True), key=lambda x: -x[1])
    summary = {
        "sample_size": len(sample),
        "top_features": [{"feature": f, "mean_abs_shap": v} for f, v in ranking[:20]],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
