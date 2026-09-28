"""Export the registry's `demo` model as a serving bundle, and optionally push it to the HF Hub.

Pulls the aliased model version from the MLflow registry, reads its lineage from the run's tags, adds
the demo's city picker and real example rows, renders the model card, and writes a bundle directory
that `fraud.serving.load_model` reads (see `fraud.serving.model` for the layout). The Space downloads
that bundle at startup, so it needs no DVC or MLflow credentials.

Usage:
    uv run python -m fraud.serving.export                # build data/bundle locally
    uv run python -m fraud.serving.export --push         # ...and upload it to the HF Hub model repo
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv

from fraud.data.paths import INTERIM_PATH, PROCESSED_DIR
from fraud.models.register import REGISTERED_MODEL_NAME
from fraud.params import REPO_ROOT, load_params
from fraud.serving.model import MODEL_CARD_FILE, write_bundle
from fraud.serving.schema import REQUIRED_COLUMNS

BUNDLE_DIR = REPO_ROOT / "data" / "bundle"
CARD_PATH = REPO_ROOT / "docs" / "model_cards" / "v0.4-demo.md"
TEST_EVALUATION_PATH = REPO_ROOT / "reports" / "test_evaluation.json"
LINEAGE_TAGS = [
    "git_commit",
    "dataset_name",
    "dataset_version",
    "dvc_data_md5",
    "pipeline_version",
    "split_spec",
]
EXAMPLE_COLUMNS = [*REQUIRED_COLUMNS, "city", "is_fraud"]


def build_cities(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per (city, state) for the demo's city pickers: label, centre point and population."""
    grouped = transactions.groupby(["city", "state"], as_index=False).agg(
        lat=("lat", "median"), long=("long", "median"), city_pop=("city_pop", "first")
    )
    grouped.insert(0, "label", grouped["city"] + ", " + grouped["state"])
    return grouped.sort_values("label").reset_index(drop=True)


def sample_examples(test: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """A seeded random sample of real test rows per class, so nothing is hand-picked to flatter the
    model: some frauds shown will be ones it misses."""
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in test.groupby("is_fraud")
    ]
    examples = pd.concat(parts)[EXAMPLE_COLUMNS].sort_values("is_fraud", ascending=False)
    return examples.reset_index(drop=True)


def build_metadata(run: Any, model_version: str, alias: str, threshold: float) -> dict[str, Any]:
    """Everything the About tab and the model card say, taken from the run's tags and metrics."""
    tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
    reference = json.loads(TEST_EVALUATION_PATH.read_text(encoding="utf-8"))
    return {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "alias": alias,
        "mlflow_run_id": run.info.run_id,
        **{tag: tags.get(tag, "n/a") for tag in LINEAGE_TAGS},
        "step": params["step"],
        "feature_set": params["feature_set"],
        "calibration": params["calibration"],
        "threshold": threshold,
        "validation": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        },
        "min_precision": load_params()["demo_model"]["min_precision"],
        # The single Phase 3 test evaluation belongs to the registry champion, NOT to this model.
        "reference_champion_test": {
            key: reference[key] for key in ("step", "feature_set", "precision", "recall", "pr_auc")
        }
        | {"confidence_intervals": reference["confidence_intervals"]},
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def render_model_card(meta: dict[str, Any]) -> str:
    v, champ = meta["validation"], meta["reference_champion_test"]
    ci = champ["confidence_intervals"]
    return f"""---
library_name: scikit-learn
tags:
  - tabular-classification
  - fraud-detection
  - {meta["step"]}
---

# Model card: fraud classifier, demo model (`{meta["model_name"]}` v{meta["model_version"]})

The stateless model served by the Phase 4 Gradio demo. It scores one transaction at a time from the
fields on the form; it has **no access to a card's earlier transactions**.

| | |
| --- | --- |
| Model | `{meta["step"]}`, calibrated with {meta["calibration"]} scaling |
| Feature pipeline | `{meta["pipeline_version"]}`, `{meta["feature_set"]}` (stateless: transaction, customer and geography features) |
| Decision threshold | `{meta["threshold"]}` (calibrated probability, picked on validation to meet precision >= {meta["min_precision"]:.2f}) |
| Dataset | `{meta["dataset_name"]}` `{meta["dataset_version"]}`, `dvc_data_md5` `{meta["dvc_data_md5"]}` |
| Split | {meta["split_spec"]} |
| Git commit | `{meta["git_commit"]}` |
| MLflow run | `{meta["mlflow_run_id"]}` (registry alias `{meta["alias"]}`, version {meta["model_version"]}) |
| Exported | {meta["exported_at"]} |

## Performance

On the **validation** split (Jul-Sep 2020), at the chosen threshold: precision {v["precision"]:.3f},
recall {v["recall"]:.3f}. This model has **not** been evaluated on the test split; the single test
evaluation of Phase 3 belongs to the registry champion, a different model:

| Phase 3 champion (`{champ["step"]}`, `{champ["feature_set"]}` features), test split | Value | 95% CI |
| --- | --- | --- |
| Recall | {champ["recall"]:.3f} | {ci["recall"][0]:.3f}-{ci["recall"][1]:.3f} |
| Precision | {champ["precision"]:.3f} | {ci["precision"][0]:.3f}-{ci["precision"][1]:.3f} |
| PR-AUC | {champ["pr_auc"]:.3f} | {ci["pr_auc"][0]:.3f}-{ci["pr_auc"][1]:.3f} |

The champion's precision on test misses the project's 0.50 target. Full analysis (error analysis,
fairness by age band and gender) is in the repository's `docs/model_cards/v0.3-model.md`.

## Explanations

Each score comes with the top reasons behind it: SHAP contributions summed per input feature and
shown next to the value that was submitted. They explain the model's raw score (log-odds, before
calibration): sign and ranking carry over to the probability, the magnitude does not.

## Known limitations

- **Simulated data** (Sparkov). The task is easier than real fraud detection; these numbers will not
  reproduce on real cardholder data.
- **No card history.** The best Phase 3 models use a card's recent velocity and reach ~0.99 recall
  with history (`v2` features); without it this model reaches {v["recall"]:.2f} recall on validation
  at the same precision target. The history-aware champion needs the online history store built in
  Phase 5.
- **Gender is a model input**, and the Phase 3 fairness check found recall and false-alarm-rate gaps
  between groups for the champion. This model has not had its own fairness evaluation.
- **Not for real decisions.** A portfolio demonstration, not a production fraud system.
"""


def main() -> None:
    serving = load_params()["serving"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", default=serving["registry_alias"])
    parser.add_argument("--out", type=Path, default=BUNDLE_DIR)
    parser.add_argument("--card-path", type=Path, default=CARD_PATH)
    parser.add_argument("--push", action="store_true", help="upload the bundle to the HF Hub repo")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, args.alias)
    if version.run_id is None:
        raise SystemExit(
            f"{REGISTERED_MODEL_NAME}@{args.alias} has no source run to read lineage from."
        )
    run = client.get_run(version.run_id)
    pipeline = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{args.alias}")

    params = load_params()
    metadata = build_metadata(
        run, str(version.version), args.alias, float(pipeline.named_steps["model"].threshold_)
    )
    cities = build_cities(pd.read_parquet(INTERIM_PATH))
    examples = sample_examples(
        pd.read_parquet(PROCESSED_DIR / "test.parquet"),
        serving["n_examples_per_class"],
        params["seed"],
    )
    card = render_model_card(metadata)
    write_bundle(args.out, pipeline, metadata, model_card=card, cities=cities, examples=examples)
    args.card_path.parent.mkdir(parents=True, exist_ok=True)
    args.card_path.write_text(card, encoding="utf-8")
    print(f"Bundle for {REGISTERED_MODEL_NAME} v{version.version} (@{args.alias}) -> {args.out}")
    print(f"Model card -> {args.card_path} (also {args.out / MODEL_CARD_FILE})")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.upload_folder(
            folder_path=str(args.out),
            repo_id=serving["hf_model_repo"],
            repo_type="model",
            commit_message=f"Export {REGISTERED_MODEL_NAME} v{version.version} (@{args.alias})",
        )
        print(f"Pushed to https://huggingface.co/{serving['hf_model_repo']}")


if __name__ == "__main__":
    main()
