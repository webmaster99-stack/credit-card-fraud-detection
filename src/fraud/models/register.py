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
    result = mlflow.register_model(f"runs:/{args.run_id}/model", REGISTERED_MODEL_NAME)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, args.alias, result.version)
    print(f"{REGISTERED_MODEL_NAME} v{result.version} -> @{args.alias} (run {args.run_id})")


if __name__ == "__main__":
    main()
