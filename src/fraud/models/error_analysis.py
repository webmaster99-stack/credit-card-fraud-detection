"""Phase 3 explainability: error analysis (missed frauds / false alarms) and a fairness check
(recall and false-alarm rate by age band and gender) on the already-computed test-set predictions.

Read-only: scores the champion bundle on test with its fixed threshold and reports on the result. It
does not retrain, retune, or otherwise touch the model; it may be re-run freely once the single test
evaluation (`evaluate_test.py`) has happened, since it changes nothing about the champion itself.

Usage: uv run python -m fraud.models.error_analysis
"""

import json
from typing import Any, cast

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud.data.paths import PROCESSED_DIR
from fraud.features.pipeline import transform_with_context
from fraud.models.train import MODEL_PATH
from fraud.params import REPO_ROOT, load_params

REPORTS_DIR = REPO_ROOT / "reports"
ERROR_SAMPLE_PATH = REPORTS_DIR / "error_analysis.json"
FAIRNESS_PATH = REPORTS_DIR / "fairness.json"

AGE_BINS = [0, 25, 35, 45, 55, 65, 120]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
DISPLAY_COLUMNS = ["trans_ts", "category", "amt", "gender", "state", "city_pop"]


def score_test(
    pipeline: Pipeline, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.Series, float]:
    feature_pipeline = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]
    context = pd.concat([train, valid], ignore_index=True)
    X_test = transform_with_context(feature_pipeline, test, context=context)
    score = pd.Series(model.predict_proba(X_test)[:, 1], index=test.index)
    return score, float(model.threshold_)


def group_metrics(
    y_true: pd.Series, y_pred: pd.Series, group: pd.Series
) -> dict[str, dict[str, Any]]:
    """Recall (among actual fraud) and false-alarm rate (among actual legit) per group value."""
    out: dict[str, dict[str, Any]] = {}
    for value in sorted(group.dropna().unique(), key=str):
        mask = group == value
        fraud_mask, legit_mask = mask & (y_true == 1), mask & (y_true == 0)
        out[str(value)] = {
            "n": int(mask.sum()),
            "n_fraud": int(fraud_mask.sum()),
            "recall": float(y_pred[fraud_mask].mean()) if fraud_mask.any() else None,
            "false_alarm_rate": float(y_pred[legit_mask].mean()) if legit_mask.any() else None,
        }
    return out


def sample_errors(test: pd.DataFrame, seed: int, k: int = 20) -> dict[str, Any]:
    """Up to k missed frauds and k false alarms, as plain records for a written write-up."""
    fn = test[(test["is_fraud"] == 1) & (test["predicted"] == 0)]
    fp = test[(test["is_fraud"] == 0) & (test["predicted"] == 1)]

    def records(df: pd.DataFrame) -> list[dict[str, Any]]:
        sample = df.sample(n=min(k, len(df)), random_state=seed) if len(df) else df
        cols = [c for c in [*DISPLAY_COLUMNS, "fraud_score"] if c in sample.columns]
        out = sample[cols].copy()
        out["trans_ts"] = out["trans_ts"].astype(str)
        return [cast(dict[str, Any], record) for record in out.to_dict(orient="records")]

    return {
        "n_missed_frauds": int(len(fn)),
        "n_false_alarms": int(len(fp)),
        "missed_fraud_sample": records(fn),
        "false_alarm_sample": records(fp),
    }


def main() -> None:
    params = load_params()
    pipeline = joblib.load(MODEL_PATH)
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet").reset_index(drop=True)

    score, threshold = score_test(pipeline, train, valid, test)
    test["fraud_score"] = score
    test["predicted"] = (score >= threshold).astype(int)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    error_report = sample_errors(test, params["seed"])
    ERROR_SAMPLE_PATH.write_text(
        json.dumps(error_report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )

    age_years = (test["trans_ts"] - test["dob"]).dt.days / 365.25
    age_band = pd.cut(age_years, bins=AGE_BINS, labels=AGE_LABELS)
    fairness = {
        "by_age_band": group_metrics(test["is_fraud"], test["predicted"], age_band),
        "by_gender": group_metrics(test["is_fraud"], test["predicted"], test["gender"]),
    }
    FAIRNESS_PATH.write_text(json.dumps(fairness, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(fairness, indent=2))


if __name__ == "__main__":
    main()
