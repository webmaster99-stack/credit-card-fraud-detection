"""The SINGLE ULB test-set evaluation, mirroring `fraud.models.evaluate_test`.

Scores the `ulb_train` bundle on the test split (latest hours) at its already-fixed validation
threshold, with bootstrap confidence intervals. Deliberately not a dvc.yaml stage.

Usage: uv run python -m fraud.ulb.evaluate_test --i-understand-this-runs-once
"""

import argparse
import json
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud.models.lineage import start_run
from fraud.models.metrics import bootstrap_ci, score_at_threshold
from fraud.params import load_params
from fraud.ulb.data import TARGET
from fraud.ulb.paths import MODEL_PATH, PROCESSED_DIR, TEST_EVALUATION_PATH
from fraud.ulb.train import lineage_overrides


def evaluate_on_test(
    pipeline: Pipeline,
    test: pd.DataFrame,
    fn_cost: float,
    fp_cost: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    y_test = test[TARGET].to_numpy()
    model = pipeline.named_steps["model"]
    score = pipeline.predict_proba(test.drop(columns=TARGET))[:, 1]
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
        help="Confirms you mean to score the ULB test split now. Used once, ever.",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    params = load_params()
    ulb = params["ulb"]
    with start_run(
        "ulb-test-evaluation",
        experiment_name=ulb["mlflow"]["experiment_name"],
        lineage_overrides=lineage_overrides(params),
    ):
        pipeline = joblib.load(MODEL_PATH)
        test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
        result = evaluate_on_test(
            pipeline, test, ulb["fn_cost"], ulb["fp_cost"], args.n_boot, params["seed"]
        )
        TEST_EVALUATION_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, int | float)})
        mlflow.log_artifact(str(TEST_EVALUATION_PATH))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
