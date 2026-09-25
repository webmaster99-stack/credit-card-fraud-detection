"""Phase 0 smoke test: a trivial MLflow run that proves lineage tags reach the server.

Usage: uv run python -m fraud.models.smoke_run
"""

import mlflow

from fraud.models.lineage import start_run


def main() -> None:
    with start_run("phase0-smoke") as run:
        mlflow.log_metric("smoke", 1.0)
        print(f"Logged run {run.info.run_id}")


if __name__ == "__main__":
    main()
