"""dvc.yaml `evaluate` stage: score the `train`-stage model bundle on the validation split.

Loads `data/models/pipeline.joblib` (features + model, already calibrated with its threshold set)
and reports precision/recall/PR-AUC/expected cost at that threshold, plus the PR curve for plotting.
Deliberately reads validation only: the test split is never touched here (see `evaluate_test.py`).

Usage: uv run python -m fraud.models.evaluate
"""

import json
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

from fraud.data.paths import PROCESSED_DIR
from fraud.features.pipeline import transform_with_context
from fraud.models.metrics import score_at_threshold
from fraud.models.train import MODEL_PATH
from fraud.params import REPO_ROOT, load_params

REPORTS_DIR = REPO_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "evaluate_metrics.json"
PR_CURVE_PATH = REPORTS_DIR / "pr_curve.json"


def evaluate_on_valid(
    full_pipeline: Pipeline,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    fn_cost: float,
    fp_cost: float,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Score `full_pipeline` (already fit, calibrated and thresholded) on validation, with train as
    card-history context. Returns (metrics at the pipeline's own threshold, the full PR curve)."""
    feature_pipeline = full_pipeline.named_steps["features"]
    model = full_pipeline.named_steps["model"]
    y_valid = valid["is_fraud"].to_numpy()

    X_valid = transform_with_context(feature_pipeline, valid, context=train)
    score = model.predict_proba(X_valid)[:, 1]

    metrics = score_at_threshold(y_valid, score, model.threshold_, fn_cost, fp_cost)
    precision, recall, thresholds = precision_recall_curve(y_valid, score)
    curve = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }
    return metrics, curve


def main() -> None:
    params = load_params()
    full_pipeline = joblib.load(MODEL_PATH)
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")

    metrics, curve = evaluate_on_valid(
        full_pipeline, train, valid, params["model"]["fn_cost"], params["model"]["fp_cost"]
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    PR_CURVE_PATH.write_text(json.dumps(curve) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
