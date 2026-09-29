"""Export a registry model (the `demo` alias, or the `champion`) as a serving bundle, and optionally
push it to the HF Hub.

Pulls the aliased model version from the MLflow registry, reads its lineage from the run's tags, adds
the demo's city picker and real example rows, renders the model card, and writes a bundle directory
that `fraud.serving.load_model` reads (see `fraud.serving.model` for the layout). The Space (demo) or
the FastAPI service (champion) downloads that bundle at startup, so neither needs DVC or MLflow
credentials.

The champion (`--alias champion`) is a v2, stateful pipeline: unlike the demo, its bundle needs the
online card-history store (`api/`) to serve correctly, and it is pushed to its own HF Hub branch
(`--revision`, default `champion`) rather than `main`, so the two bundles never collide.

Usage:
    uv run python -m fraud.serving.export                          # build data/bundle locally
    uv run python -m fraud.serving.export --push                   # ...and push it (branch `main`)
    uv run python -m fraud.serving.export --alias champion --push  # champion -> branch `champion`
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
CHAMPION_CARD_PATH = REPO_ROOT / "docs" / "model_cards" / "v1.0-api.md"
TEST_EVALUATION_PATH = REPO_ROOT / "reports" / "test_evaluation.json"
# Which params.yaml section defines an alias's min_precision budget (model lineage table).
ALIAS_PARAM_SECTION = {"demo": "demo_model", "champion": "model", "challenger": "model"}
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
        "min_precision": load_params()[ALIAS_PARAM_SECTION[alias]]["min_precision"],
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


def build_champion_metadata(
    run: Any, model_version: str, alias: str, threshold: float
) -> dict[str, Any]:
    """Like `build_metadata`, but for the v2 champion/challenger the API serves: its own test-set
    numbers (not a "reference" to a different model), and no city/example data dependency on this
    being the stateless demo. Fails loudly if `reports/test_evaluation.json` belongs to some other
    model - Phase 3's single test evaluation is tied to one specific run, not to whichever model
    happens to hold the alias today."""
    tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
    test = json.loads(TEST_EVALUATION_PATH.read_text(encoding="utf-8"))
    if (test["step"], test["feature_set"]) != (params["step"], params["feature_set"]):
        raise SystemExit(
            f"reports/test_evaluation.json is for {test['step']}/{test['feature_set']}, but "
            f"{REGISTERED_MODEL_NAME}@{alias} is {params['step']}/{params['feature_set']}. "
            "Re-run the Phase 3 test evaluation for this model before exporting it."
        )
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
        "validation": {"precision": metrics["precision"], "recall": metrics["recall"]},
        "min_precision": load_params()[ALIAS_PARAM_SECTION[alias]]["min_precision"],
        "test": {key: test[key] for key in ("precision", "recall", "pr_auc")}
        | {"confidence_intervals": test["confidence_intervals"]},
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def render_champion_model_card(meta: dict[str, Any]) -> str:
    v, t = meta["validation"], meta["test"]
    ci = t["confidence_intervals"]
    return f"""---
library_name: scikit-learn
tags:
  - tabular-classification
  - fraud-detection
  - {meta["step"]}
---

# Model card: fraud classifier, {meta["alias"]} (`{meta["model_name"]}` v{meta["model_version"]})

The stateful (v2) model served by the Phase 5 API. Unlike the demo, it uses a card's recent
transaction history and **requires the online history store** (`api/`); scoring it directly from a
bare transaction, with no history, does not reproduce these numbers.

| | |
| --- | --- |
| Model | `{meta["step"]}`, calibrated with {meta["calibration"]} scaling |
| Feature pipeline | `{meta["pipeline_version"]}`, `{meta["feature_set"]}` (stateful: adds card velocity and behavioural features) |
| Decision threshold | `{meta["threshold"]}` (calibrated probability, picked on validation to meet precision >= {meta["min_precision"]:.2f}) |
| Dataset | `{meta["dataset_name"]}` `{meta["dataset_version"]}`, `dvc_data_md5` `{meta["dvc_data_md5"]}` |
| Split | {meta["split_spec"]} |
| Git commit | `{meta["git_commit"]}` |
| MLflow run | `{meta["mlflow_run_id"]}` (registry alias `{meta["alias"]}`, version {meta["model_version"]}) |
| Exported | {meta["exported_at"]} |

## Performance

On **validation**, at the chosen threshold: precision {v["precision"]:.3f}, recall {v["recall"]:.3f}.
On the **test** split (touched once, Phase 3 protocol step 5):

| Test split | Value | 95% CI |
| --- | --- | --- |
| Recall | {t["recall"]:.3f} | {ci["recall"][0]:.3f}-{ci["recall"][1]:.3f} |
| Precision | {t["precision"]:.3f} | {ci["precision"][0]:.3f}-{ci["precision"][1]:.3f} |
| PR-AUC | {t["pr_auc"]:.3f} | {ci["pr_auc"][0]:.3f}-{ci["pr_auc"][1]:.3f} |

Precision on test misses the project's 0.50 target - a real, modestly-sized generalization gap,
documented rather than fixed by re-tuning against test. Full analysis (error analysis, fairness by
age band and gender) is in `docs/model_cards/v0.3-model.md`.

## Explanations

Each score comes with the top reasons behind it: SHAP contributions summed per input feature and
shown next to the value that was submitted. They explain the model's raw score (log-odds, before
calibration): sign and ranking carry over to the probability, the magnitude does not.

## Known limitations

- **Simulated data** (Sparkov). The task is easier than real fraud detection; these numbers will not
  reproduce on real cardholder data.
- **Needs history.** Scores depend on the requesting card's stored transactions; a card with no
  history yet (or a history store that has just started serving) scores as if it were a first-ever
  transaction on that card.
- **Gender is a model input**, and the Phase 3 fairness check found recall and false-alarm-rate gaps
  between groups. See `docs/model_cards/v0.3-model.md`.
- **Not for real decisions.** A portfolio demonstration, not a production fraud system.
"""


def main() -> None:
    serving = load_params()["serving"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", default=serving["registry_alias"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--card-path", type=Path, default=None)
    parser.add_argument("--push", action="store_true", help="upload the bundle to the HF Hub repo")
    parser.add_argument(
        "--revision",
        default=None,
        help="HF Hub branch to push to (default: main for the demo "
        "alias, the alias itself otherwise)",
    )
    args = parser.parse_args()
    is_demo = args.alias == "demo"
    out_dir = args.out or (BUNDLE_DIR if is_demo else REPO_ROOT / "data" / f"bundle_{args.alias}")
    card_path = args.card_path or (CARD_PATH if is_demo else CHAMPION_CARD_PATH)
    revision = args.revision or ("main" if is_demo else args.alias)

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
    threshold = float(pipeline.named_steps["model"].threshold_)
    if is_demo:
        metadata = build_metadata(run, str(version.version), args.alias, threshold)
        card = render_model_card(metadata)
    else:
        metadata = build_champion_metadata(run, str(version.version), args.alias, threshold)
        card = render_champion_model_card(metadata)

    cities = build_cities(pd.read_parquet(INTERIM_PATH))
    examples = sample_examples(
        pd.read_parquet(PROCESSED_DIR / "test.parquet"),
        serving["n_examples_per_class"],
        params["seed"],
    )
    write_bundle(out_dir, pipeline, metadata, model_card=card, cities=cities, examples=examples)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")
    print(f"Bundle for {REGISTERED_MODEL_NAME} v{version.version} (@{args.alias}) -> {out_dir}")
    print(f"Model card -> {card_path} (also {out_dir / MODEL_CARD_FILE})")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        if revision != "main":
            api.create_branch(
                serving["hf_model_repo"], branch=revision, repo_type="model", exist_ok=True
            )
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=serving["hf_model_repo"],
            repo_type="model",
            revision=revision,
            commit_message=f"Export {REGISTERED_MODEL_NAME} v{version.version} (@{args.alias})",
        )
        print(f"Pushed to https://huggingface.co/{serving['hf_model_repo']}/tree/{revision}")


if __name__ == "__main__":
    main()
