"""dvc.yaml `ulb_train` stage: the ladder rungs on ULB, same protocol as Sparkov.

For each configured step: fit on train, calibrate on validation, pick the threshold on validation
(recall at precision >= min_precision). The best step by that metric is saved as the deployable
Pipeline and, with --register, registered as `fraud-ulb`, never under the Sparkov model's name.
This stage never reads the test split; `fraud.ulb.evaluate_test` does, once.

Usage: uv run python -m fraud.ulb.train [--register]
"""

import argparse
import json
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud.models.estimator import CalibratedThresholdClassifier
from fraud.models.ladder import build_estimator
from fraud.models.lineage import dvc_data_md5, start_run
from fraud.models.metrics import pr_auc, recall_at_precision
from fraud.models.register import register_run_model
from fraud.params import REPO_ROOT, load_params
from fraud.ulb import PIPELINE_VERSION
from fraud.ulb.data import TARGET, split_spec_text
from fraud.ulb.features import build_pipeline, feature_list
from fraud.ulb.paths import LADDER_PATH, MODEL_PATH, PROCESSED_DIR

DVC_DATA_PATH = "data/processed_ulb"


def lineage_overrides(params: dict[str, Any]) -> dict[str, str]:
    ulb = params["ulb"]
    return {
        "dataset_name": ulb["dataset"]["name"],
        "dataset_version": ulb["dataset"]["version"],
        "dvc_data_md5": dvc_data_md5(REPO_ROOT / "dvc.lock", DVC_DATA_PATH),
        "pipeline_version": PIPELINE_VERSION,
        "split_spec": split_spec_text(ulb["split"]),
    }


def fit_step(
    step: str,
    hyperparams: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    ulb_cfg: dict[str, Any],
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit -> calibrate -> threshold on the splits given. No file or MLflow I/O."""
    feature_pipeline = build_pipeline(ulb_cfg["features"])
    model = CalibratedThresholdClassifier(
        build_estimator(step, hyperparams, seed), ulb_cfg["calibration"]
    )
    pipeline = Pipeline([("features", feature_pipeline), ("model", model)])
    X_train, X_valid_raw = train.drop(columns=TARGET), valid.drop(columns=TARGET)
    pipeline.fit(X_train, train[TARGET].to_numpy())
    y_valid = valid[TARGET].to_numpy()
    X_valid = feature_pipeline.transform(X_valid_raw)
    model.calibrate(X_valid, y_valid)
    score = model.predict_proba(X_valid)[:, 1]
    result = recall_at_precision(y_valid, score, ulb_cfg["min_precision"])
    model.set_threshold(result.threshold)
    return pipeline, {
        "step": step,
        "threshold": result.threshold,
        "precision": result.precision,
        "recall": result.recall,
        "met_budget": result.met_budget,
        "pr_auc": pr_auc(y_valid, score),
    }


def best_step(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Best validation recall among steps that met the precision budget (all steps if none did)."""
    eligible = [r for r in results if r["met_budget"]] or results
    return max(eligible, key=lambda r: (r["recall"], r["pr_auc"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="register the best step")
    register = parser.parse_args().register

    params = load_params()
    ulb = params["ulb"]
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")

    results: list[dict[str, Any]] = []
    pipelines: dict[str, Pipeline] = {}
    ignore = (LADDER_PATH.relative_to(REPO_ROOT).as_posix(),)
    with start_run(
        "ulb-ladder",
        ignore_dirty=ignore,
        experiment_name=ulb["mlflow"]["experiment_name"],
        lineage_overrides=lineage_overrides(params),
    ) as parent:
        for step, hyperparams in ulb["steps"].items():
            with mlflow.start_run(run_name=f"ulb-{step}", nested=True):
                pipeline, metrics = fit_step(step, hyperparams, train, valid, ulb, params["seed"])
                mlflow.log_params({"step": step, **{f"hp_{k}": v for k, v in hyperparams.items()}})
                mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
                )
            results.append(metrics)
            pipelines[step] = pipeline
            print(json.dumps(metrics))

        best = best_step(results)
        winner = pipelines[best["step"]]
        mlflow.set_tag("best_step", best["step"])
        mlflow.log_metrics({f"best_{k}": float(best[k]) for k in ("precision", "recall", "pr_auc")})
        example = train.drop(columns=TARGET).iloc[:5]
        features = winner.named_steps["features"]
        mlflow.log_dict(feature_list(features, features.transform(example)), "feature_list.json")
        mlflow.sklearn.log_model(
            winner,
            name="model",
            signature=mlflow.models.infer_signature(example, winner.predict_proba(example)),
            input_example=example,
            serialization_format="cloudpickle",
        )
        run_id = parent.info.run_id

    if register:
        register_run_model(run_id, ulb["registry_name"], "champion")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner, MODEL_PATH)
    LADDER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LADDER_PATH.write_text(
        json.dumps({"best": best["step"], "steps": results}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"best: {best['step']} (recall {best['recall']:.3f} @ precision {best['precision']:.3f})")


if __name__ == "__main__":
    main()
