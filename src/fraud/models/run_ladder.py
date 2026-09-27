"""Run the whole Phase 3 model ladder sweep (docs/plan.md) in one process: build the feature
pipeline once, tune every rung on the same train/validation split, and record results to
reports/model_ladder.json. One MLflow run stamped with lineage tags wraps the whole sweep; each
step's sweep is a child run, each Optuna trial a grandchild (see docs/infra.md's DagsHub run count).

Usage:
    uv run python -m fraud.models.run_ladder --feature-set v1
    uv run python -m fraud.models.run_ladder --feature-set v1 --steps logreg lightgbm
"""

import argparse

import pandas as pd

from fraud.data.paths import PROCESSED_DIR
from fraud.features.pipeline import build_pipeline, transform_with_context
from fraud.models.ladder import LADDER
from fraud.models.lineage import start_run
from fraud.models.tune import tune_step
from fraud.params import load_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--steps", nargs="*", default=None, help="ladder step keys to run; default: all"
    )
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    params = load_params()
    n_trials = args.n_trials or params["tune"]["n_trials"]
    steps = args.steps or [step.key for step in LADDER]

    with start_run(f"ladder-sweep-{args.feature_set}"):
        train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
        valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")
        y_train, y_valid = train["is_fraud"].to_numpy(), valid["is_fraud"].to_numpy()

        pipeline = build_pipeline(args.feature_set, params["features"])
        X_train = pipeline.fit_transform(train)
        X_valid = transform_with_context(pipeline, valid, context=train)

        for key in steps:
            print(f"=== tuning {key} ({args.feature_set}) ===", flush=True)
            result = tune_step(
                key,
                X_train,
                y_train,
                X_valid,
                y_valid,
                params["model"]["min_precision"],
                n_trials,
                params["seed"],
                args.feature_set,
                nested=True,
            )
            print(result, flush=True)


if __name__ == "__main__":
    main()
