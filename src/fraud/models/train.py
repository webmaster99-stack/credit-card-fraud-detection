"""dvc.yaml `train` stage: fit the ladder step pinned in params.yaml's `model` section on the
training split, calibrate and pick a threshold on validation, and save the deployable Pipeline
(features + model, per CLAUDE.md's model lineage rules).

The full Phase 3 ladder sweep (docs/plan.md) is driven by `tune.py`, logging one MLflow parent run
per step with nested trials; this script reproduces just the winning configuration deterministically
so `dvc repro train` always rebuilds the same artifact from the same inputs. It never reads the test
split: that happens once, at the end of Phase 3, via `fraud.models.evaluate_test`.

Usage: uv run python -m fraud.models.train [--section model|demo_model]
"""

import argparse
import json
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud.data.paths import PROCESSED_DIR
from fraud.features.pipeline import build_pipeline, feature_list, transform_with_context
from fraud.models.estimator import CalibratedThresholdClassifier
from fraud.models.ladder import build_estimator
from fraud.models.lineage import start_run
from fraud.models.metrics import recall_at_precision
from fraud.params import REPO_ROOT, load_params

MODEL_DIR = REPO_ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
REPORTS_DIR = REPO_ROOT / "reports"
TRAIN_METRICS_PATH = REPORTS_DIR / "train_metrics.json"

# Outputs per params.yaml section: `model` is the Phase 3 champion candidate, `demo_model` the v1
# stateless model the Phase 4 demo serves.
DEMO_MODEL_PATH = MODEL_DIR / "demo_pipeline.joblib"
DEMO_TRAIN_METRICS_PATH = REPORTS_DIR / "train_metrics_demo.json"
SECTION_OUTPUTS = {
    "model": (MODEL_PATH, TRAIN_METRICS_PATH),
    "demo_model": (DEMO_MODEL_PATH, DEMO_TRAIN_METRICS_PATH),
}


def fit_champion_candidate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    model_cfg: dict[str, Any],
    features_cfg: dict[str, Any],
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit -> calibrate -> pick threshold, all on the splits given. No MLflow or file I/O here, so
    it can be reused directly by tests and by `evaluate_test.py`."""
    y_train, y_valid = train["is_fraud"].to_numpy(), valid["is_fraud"].to_numpy()
    feature_pipeline = build_pipeline(model_cfg["feature_set"], features_cfg)
    estimator = build_estimator(model_cfg["step"], model_cfg["hyperparams"], seed)
    model = CalibratedThresholdClassifier(estimator, model_cfg["calibration"])
    full_pipeline = Pipeline([("features", feature_pipeline), ("model", model)])

    full_pipeline.fit(train, y_train)
    # `feature_pipeline` is fitted in place by the Pipeline above; reuse it so history features see
    # train as context, exactly as `fraud.features.build` does for the `features` dvc stage.
    X_valid = transform_with_context(feature_pipeline, valid, context=train)
    model.calibrate(X_valid, y_valid)
    calibrated_valid_score = model.predict_proba(X_valid)[:, 1]
    threshold_result = recall_at_precision(
        y_valid, calibrated_valid_score, model_cfg["min_precision"]
    )
    model.set_threshold(threshold_result.threshold)

    metrics = {
        "step": model_cfg["step"],
        "feature_set": model_cfg["feature_set"],
        "calibration": model_cfg["calibration"],
        "threshold": threshold_result.threshold,
        "precision": threshold_result.precision,
        "recall": threshold_result.recall,
        "met_budget": threshold_result.met_budget,
    }
    return full_pipeline, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=list(SECTION_OUTPUTS), default="model")
    section = parser.parse_args().section
    model_path, metrics_path = SECTION_OUTPUTS[section]

    params = load_params()
    model_cfg = params[section]

    # The clean-tree check must run before this script writes anything of its own (the model
    # bundle, the train metrics report): otherwise its own output would make the tree "dirty" for
    # the very next run.
    run_name = f"train-{model_cfg['step']}-{model_cfg['feature_set']}"
    ignore_dirty = (metrics_path.relative_to(REPO_ROOT).as_posix(),)
    with start_run(run_name, ignore_dirty=ignore_dirty) as run:
        mlflow.set_tag("params_section", section)
        train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
        valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")

        full_pipeline, metrics = fit_champion_candidate(
            train, valid, model_cfg, params["features"], params["seed"]
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(full_pipeline, model_path)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

        mlflow.log_params(
            {
                "step": model_cfg["step"],
                "feature_set": model_cfg["feature_set"],
                "calibration": model_cfg["calibration"],
                **{f"hp_{k}": v for k, v in model_cfg["hyperparams"].items()},
            }
        )
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, int | float) and k != "met_budget"}
        )
        mlflow.log_metric("met_budget", float(metrics["met_budget"]))
        example = train.iloc[:2]
        features = full_pipeline.named_steps["features"]
        mlflow.log_dict(
            feature_list(features, features.transform(example), model_cfg["feature_set"]),
            "feature_list.json",
        )
        signature = mlflow.models.infer_signature(example, full_pipeline.predict_proba(example))
        mlflow.sklearn.log_model(
            full_pipeline,
            name="model",
            signature=signature,
            input_example=example,
            # skops (the mlflow 3.x default) refuses custom classes like ours without an explicit
            # trusted-types allowlist; cloudpickle round-trips them with no extra bookkeeping.
            serialization_format="cloudpickle",
        )
    print(json.dumps(metrics, indent=2))
    print(f"MLflow run id: {run.info.run_id}")


if __name__ == "__main__":
    main()
