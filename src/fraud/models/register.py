"""Register a logged MLflow run's model in the registry and set its alias, per CLAUDE.md's model
lineage rules ("Model Registry (aliases `champion`, `challenger`)").

Usage: uv run python -m fraud.models.register <run_id> <champion|challenger|demo>

`demo` marks the v1 stateless model the Phase 4 demo serves; it never moves champion/challenger.
"""

import argparse

import mlflow
from dotenv import load_dotenv

from fraud.params import REPO_ROOT

REGISTERED_MODEL_NAME = "fraud-classifier"


def register_run_model(
    run_id: str, name: str = REGISTERED_MODEL_NAME, alias: str = "champion"
) -> None:
    """Register the model logged by `run_id` under `name` and point `alias` at the new version."""
    load_dotenv(REPO_ROOT / ".env")
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    if not run.outputs or not run.outputs.model_outputs:
        raise SystemExit(f"Run {run_id} has no logged model outputs.")
    # MLflow 3.x logs a model as its own "Logged Model" entity; models:/<model_id> resolves it
    # directly and sidesteps runs:/<run_id>/<name> lookup quirks against non-Databricks backends.
    model_id = run.outputs.model_outputs[0].model_id
    result = mlflow.register_model(f"models:/{model_id}", name)
    client.set_registered_model_alias(name, alias, result.version)
    print(f"{name} v{result.version} -> @{alias} (run {run_id})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("alias", choices=["champion", "challenger", "demo"])
    args = parser.parse_args()
    register_run_model(args.run_id, REGISTERED_MODEL_NAME, args.alias)


if __name__ == "__main__":
    main()
