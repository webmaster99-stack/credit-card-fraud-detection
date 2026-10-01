"""Phase 3 protocol step 5: the SINGLE test-set evaluation.

Scores the `train`-stage model bundle on the test split with its already-fixed threshold (picked on
validation, never touched here) and reports bootstrap confidence intervals. Deliberately **not** a
dvc.yaml stage: `dvc repro` must never be able to touch the test split as a side effect of an
unrelated change. This script is meant to be run once, ever, for a given champion.

Usage: uv run python -m fraud.models.evaluate_test --i-understand-this-runs-once
"""

import argparse
import json
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud.data.paths import PROCESSED_DIR
from fraud.features.pipeline import transform_with_context
from fraud.models.lineage import start_run
from fraud.models.metrics import bootstrap_ci, score_at_threshold
from fraud.models.train import MODEL_PATH
from fraud.params import REPO_ROOT, load_params

REPORTS_DIR = REPO_ROOT / "reports"
TEST_METRICS_PATH = REPORTS_DIR / "test_evaluation.json"


def evaluate_on_test(
    full_pipeline: Pipeline,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    fn_cost: float,
    fp_cost: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Score `full_pipeline` on test, with everything earlier (train + valid) as card-history
    context, at the pipeline's own fixed threshold. Adds bootstrap CIs for precision/recall/PR-AUC.
    """
    feature_pipeline = full_pipeline.named_steps["features"]
    model = full_pipeline.named_steps["model"]
    y_test = test["is_fraud"].to_numpy()

    context = pd.concat([train, valid], ignore_index=True)
    X_test = transform_with_context(feature_pipeline, test, context=context)
    score = model.predict_proba(X_test)[:, 1]

    metrics = score_at_threshold(y_test, score, model.threshold_, fn_cost, fp_cost)
    ci = bootstrap_ci(y_test, score, model.threshold_, fn_cost, fp_cost, n_boot=n_boot, seed=seed)
    return {**metrics, "confidence_intervals": {k: list(v) for k, v in ci.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-understand-this-runs-once",
        dest="confirmed",
        action="store_true",
        required=True,
        help="Confirms you mean to score the test split now. It is used once, ever, per champion.",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    del args.confirmed  # only exists to make this an explicit, deliberate flag

    params = load_params()
    model_cfg = params["model"]

    with start_run("test-evaluation"):
        full_pipeline = joblib.load(MODEL_PATH)
        train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
        valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")
        test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

        result = evaluate_on_test(
            full_pipeline,
            train,
            valid,
            test,
            model_cfg["fn_cost"],
            model_cfg["fp_cost"],
            args.n_boot,
            params["seed"],
        )
        result = {"step": model_cfg["step"], "feature_set": model_cfg["feature_set"], **result}

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        TEST_METRICS_PATH.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, int | float)})
        mlflow.log_artifact(str(TEST_METRICS_PATH))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
