"""Load the deployable bundle and score with it. Gradio and FastAPI both go through here, so they
can never disagree about a prediction.

A bundle is a directory (or a Hugging Face Hub model repo with the same layout):

    pipeline.joblib    the fitted scikit-learn Pipeline (features + calibrated model + threshold)
    metadata.json      model/pipeline version, lineage tags, threshold, validation metrics
    feature_list.json  final feature columns and dtypes
    README.md          model card
    template.csv       the documented input schema, with two example rows
    cities.csv         city picker for the demo form (label, city, state, lat, long, city_pop)
    examples.csv       real rows from the test split for "Try an example" (input columns + is_fraud)

It needs no DVC or MLflow access.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud.explain.shap_values import build_explainer, compute_shap_values
from fraud.features.pipeline import feature_list, transform_with_context
from fraud.params import load_params
from fraud.serving.explain import Reason, build_reasons
from fraud.serving.schema import template_frame, validate_transactions

PIPELINE_FILE = "pipeline.joblib"
METADATA_FILE = "metadata.json"
FEATURE_LIST_FILE = "feature_list.json"
MODEL_CARD_FILE = "README.md"
TEMPLATE_FILE = "template.csv"
CITIES_FILE = "cities.csv"
EXAMPLES_FILE = "examples.csv"

REQUIRED_METADATA = ["model_name", "model_version", "pipeline_version", "feature_set", "threshold"]


@dataclass
class ServingModel:
    pipeline: Pipeline
    metadata: dict[str, Any]
    bundle_dir: Path | None = None

    @property
    def feature_pipeline(self) -> Pipeline:
        step: Pipeline = self.pipeline.named_steps["features"]
        return step

    @property
    def classifier(self) -> Any:
        return self.pipeline.named_steps["model"]

    @property
    def threshold(self) -> float:
        return float(self.classifier.threshold_)

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def pipeline_version(self) -> str:
        return str(self.metadata["pipeline_version"])

    @cached_property
    def explainer(self) -> Any:
        return build_explainer(self.classifier.estimator)

    def read_table(self, filename: str) -> pd.DataFrame:
        """A CSV shipped in the bundle (cities, examples, template)."""
        if self.bundle_dir is None:
            raise FileNotFoundError("This model was not loaded from a bundle directory.")
        return pd.read_csv(self.bundle_dir / filename)


def load_model(source: str | Path | None = None, revision: str | None = None) -> ServingModel:
    """Load a bundle from a local directory, or download one from a Hugging Face Hub model repo.

    ``source`` defaults to the repo named in params.yaml (`serving.hf_model_repo`).
    """
    source = source if source is not None else load_params()["serving"]["hf_model_repo"]
    path = Path(source)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=str(source), revision=revision))
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_METADATA if k not in metadata]
    if missing:
        raise ValueError(f"Bundle metadata is missing {missing}.")
    pipeline = joblib.load(path / PIPELINE_FILE)
    model = ServingModel(pipeline=pipeline, metadata=metadata, bundle_dir=path)
    if model.threshold != float(metadata["threshold"]):
        raise ValueError("Bundle metadata threshold does not match the pipeline's threshold.")
    return model


def write_bundle(
    out_dir: Path,
    pipeline: Pipeline,
    metadata: dict[str, Any],
    *,
    model_card: str,
    cities: pd.DataFrame,
    examples: pd.DataFrame,
) -> Path:
    """Write a bundle directory that `load_model` can read back. Returns ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    template = template_frame()
    features = pipeline.named_steps["features"]
    validated = validate_transactions(template)
    # A v2 (history) pipeline needs a card_id to group by; a placeholder is enough here since this
    # transform only discovers output column names and dtypes, not real feature values.
    probe = validated.assign(card_id=[f"template-{i}" for i in range(len(validated))])
    transformed = features.transform(probe)
    listing = feature_list(features, transformed, metadata["feature_set"])

    joblib.dump(pipeline, out_dir / PIPELINE_FILE)
    (out_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / FEATURE_LIST_FILE).write_text(
        json.dumps(listing, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / MODEL_CARD_FILE).write_text(model_card, encoding="utf-8", newline="\n")
    template.to_csv(out_dir / TEMPLATE_FILE, index=False)
    cities.to_csv(out_dir / CITIES_FILE, index=False)
    examples.to_csv(out_dir / EXAMPLES_FILE, index=False)
    return out_dir


def predict(
    model: ServingModel,
    df: pd.DataFrame,
    *,
    context: pd.DataFrame | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Score raw transactions: `fraud_probability` (calibrated) and `flagged` (>= threshold).

    ``context`` is earlier stored transactions, used only by a history (v2) pipeline. Raises
    `InputValidationError` with readable messages if ``df`` breaks the input contract.
    """
    frame = validate_transactions(df, max_rows)
    X = transform_with_context(model.feature_pipeline, frame, context)
    proba = model.classifier.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {"fraud_probability": proba, "flagged": proba >= model.threshold}, index=frame.index
    )


def explain(
    model: ServingModel,
    df: pd.DataFrame,
    *,
    top_k: int | None = None,
    context: pd.DataFrame | None = None,
) -> list[list[Reason]]:
    """The top-k reasons behind each row's score, most influential first (one list per row)."""
    k = top_k if top_k is not None else load_params()["serving"]["top_k_reasons"]
    frame = validate_transactions(df)
    X = transform_with_context(model.feature_pipeline, frame, context)
    union = model.feature_pipeline.named_steps["features"]
    raw = transform_with_context(union, frame, context)
    shap_values = compute_shap_values(model.explainer, X)
    return build_reasons(shap_values, list(X.columns), raw, k)
