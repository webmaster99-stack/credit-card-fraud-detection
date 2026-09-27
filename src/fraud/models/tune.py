"""Optuna sweep over one ladder step: fit on train, score on validation (Phase 3 protocol step 1).

Every trial is a nested MLflow run under one parent run per (step, feature_set), so a 25-trial sweep
across all tunable steps stays well inside the tracked-run counts noted in docs/infra.md.

Usage: `uv run python -m fraud.models.tune --step lightgbm --feature-set v1`
"""

import argparse
from typing import Any

import mlflow
import optuna
import pandas as pd

from fraud.models.ladder import fraud_score, get_step
from fraud.models.metrics import pr_auc, recall_at_precision
from fraud.models.results import LADDER_RESULTS_PATH, LadderResult, append_ladder_result

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(
    trial: optuna.Trial,
    step_key: str,
    X_train: pd.DataFrame,
    y_train: Any,
    X_valid: pd.DataFrame,
    y_valid: Any,
    min_precision: float,
    seed: int,
) -> float:
    step = get_step(step_key)
    params = step.space(trial) if step.space is not None else {}
    estimator = step.build(params, seed)
    with mlflow.start_run(run_name=f"{step_key}-trial{trial.number}", nested=True):
        estimator.fit(X_train, y_train)
        score = fraud_score(estimator, X_valid)
        result = recall_at_precision(y_valid, score, min_precision)
        auc = pr_auc(y_valid, score)
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "recall_at_precision": result.recall,
                "precision_at_threshold": result.precision,
                "pr_auc": auc,
                "met_budget": float(result.met_budget),
            }
        )
    trial.set_user_attr("precision", result.precision)
    trial.set_user_attr("pr_auc", auc)
    trial.set_user_attr("met_budget", result.met_budget)
    # Trials that miss the precision budget are not thrown out: their recall is reported as 0 by
    # `recall_at_precision`, so Optuna still ranks budget-meeting trials above them.
    return result.recall


def tune_step(
    step_key: str,
    X_train: pd.DataFrame,
    y_train: Any,
    X_valid: pd.DataFrame,
    y_valid: Any,
    min_precision: float,
    n_trials: int,
    seed: int,
    feature_set: str,
    results_path: Any = LADDER_RESULTS_PATH,
    nested: bool = False,
) -> LadderResult:
    """Run (or single-shot evaluate, if not tunable) one ladder step; return its best result.

    `nested=True` when called from inside another active MLflow run (`run_ladder.py`'s sweep-wide
    parent run), so this step's run becomes a child rather than erroring on an already-active run.
    """
    step = get_step(step_key)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trials = n_trials if step.space is not None else 1
    with mlflow.start_run(run_name=f"tune-{step_key}-{feature_set}", nested=nested):
        mlflow.set_tags({"ladder_step": step_key, "feature_set": feature_set})
        study.optimize(
            lambda trial: _objective(
                trial, step_key, X_train, y_train, X_valid, y_valid, min_precision, seed
            ),
            n_trials=trials,
        )
        best = study.best_trial
        assert best.value is not None
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metrics(
            {
                "best_recall_at_precision": best.value,
                "best_precision_at_threshold": best.user_attrs["precision"],
                "best_pr_auc": best.user_attrs["pr_auc"],
            }
        )
    result = LadderResult(
        step_id=step.id,
        key=step.key,
        name=step.name,
        feature_set=feature_set,
        hyperparams=best.params,
        recall_at_precision=best.value,
        precision_at_threshold=best.user_attrs["precision"],
        pr_auc=best.user_attrs["pr_auc"],
        met_budget=bool(best.user_attrs["met_budget"]),
        n_trials=trials,
    )
    append_ladder_result(result, path=results_path)
    return result


def _load_split(
    feature_set: str, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, Any, pd.DataFrame, Any]:
    from fraud.data.paths import PROCESSED_DIR
    from fraud.features.pipeline import build_pipeline, transform_with_context

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")
    pipeline = build_pipeline(feature_set, cfg)
    X_train = pipeline.fit_transform(train)
    X_valid = transform_with_context(pipeline, valid, context=train)
    return X_train, train["is_fraud"].to_numpy(), X_valid, valid["is_fraud"].to_numpy()


def main() -> None:
    from fraud.models.ladder import STEPS_BY_KEY
    from fraud.models.lineage import start_run
    from fraud.params import load_params

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=list(STEPS_BY_KEY))
    parser.add_argument("--feature-set", default="v1", choices=["v1", "v2"])
    parser.add_argument("--n-trials", type=int, default=None)
    args, _ = parser.parse_known_args()

    params = load_params()
    X_train, y_train, X_valid, y_valid = _load_split(args.feature_set, params["features"])
    n_trials = args.n_trials or params["tune"]["n_trials"]
    with start_run(f"tune-{args.step}"):
        result = tune_step(
            args.step,
            X_train,
            y_train,
            X_valid,
            y_valid,
            params["model"]["min_precision"],
            n_trials,
            params["seed"],
            args.feature_set,
        )
    print(result)


if __name__ == "__main__":
    main()
