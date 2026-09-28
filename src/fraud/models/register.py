"""Register a logged MLflow run's model in the registry and set its alias, per CLAUDE.md's model
lineage rules ("Model Registry (aliases `champion`, `challenger`)").

Usage: uv run python -m fraud.models.register <run_id> <champion|challenger>
"""

import argparse

import mlflow
from dotenv import load_dotenv

from fraud.params import REPO_ROOT

REGISTERED_MODEL_NAME = "fraud-classifier"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("alias", choices=["champion", "challenger"])
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    client = mlflow.MlflowClient()
    run = client.get_run(args.run_id)
    if not run.outputs or not run.outputs.model_outputs:
        raise SystemExit(f"Run {args.run_id} has no logged model outputs.")
    # MLflow 3.x logs a model as its own "Logged Model" entity; models:/<model_id> resolves it
    # directly and sidesteps runs:/<run_id>/<name> lookup quirks against non-Databricks backends.
    model_id = run.outputs.model_outputs[0].model_id
    result = mlflow.register_model(f"models:/{model_id}", REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, args.alias, result.version)
    print(f"{REGISTERED_MODEL_NAME} v{result.version} -> @{args.alias} (run {args.run_id})")


if __name__ == "__main__":
    main()
