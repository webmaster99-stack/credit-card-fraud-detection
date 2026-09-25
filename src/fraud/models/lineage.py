"""Stamp every MLflow run with the lineage tags required by CLAUDE.md.

A registered model version must answer four questions from its tags alone:
which code, which data, which features, which pipeline version.
"""

import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import yaml
from dotenv import load_dotenv

from fraud.features import PIPELINE_VERSION
from fraud.params import PARAMS_PATH, REPO_ROOT, load_params

DVC_DATA_PATH = "data/processed"
NOT_AVAILABLE = "n/a"


class DirtyTreeError(RuntimeError):
    """Raised when training is attempted with uncommitted changes."""


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def git_commit(repo: Path = REPO_ROOT) -> str:
    return _git("rev-parse", "--short", "HEAD", cwd=repo)


def ensure_clean_tree(repo: Path = REPO_ROOT) -> None:
    if _git("status", "--porcelain", cwd=repo):
        raise DirtyTreeError("Refusing to run: git tree has uncommitted changes.")


def dvc_data_md5(lock_path: Path = REPO_ROOT / "dvc.lock") -> str:
    """Return the md5 recorded in dvc.lock for data/processed, or 'n/a' if absent."""
    if not lock_path.exists():
        return NOT_AVAILABLE
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    for stage in (lock.get("stages") or {}).values():
        for out in stage.get("outs") or []:
            if out.get("path") == DVC_DATA_PATH:
                return str(out["md5"])
    return NOT_AVAILABLE


def split_spec(split: dict[str, str]) -> str:
    return (
        f"train {split['train_start']}..{split['train_end']}, "
        f"valid {split['valid_start']}..{split['valid_end']}, "
        f"test {split['test_start']}..{split['test_end']}"
    )


def lineage_tags(params: dict[str, Any], repo: Path = REPO_ROOT) -> dict[str, str]:
    dataset = params["dataset"]
    return {
        "git_commit": git_commit(repo),
        "dataset_name": dataset["name"],
        "dataset_version": dataset["version"],
        "dvc_data_md5": dvc_data_md5(repo / "dvc.lock"),
        "pipeline_version": PIPELINE_VERSION,
        "split_spec": split_spec(params["split"]),
    }


def _log_requirements_lock(repo: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "requirements.lock"
        with out.open("w", encoding="utf-8") as f:
            subprocess.run(
                ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project"],
                cwd=repo,
                stdout=f,
                check=True,
            )
        mlflow.log_artifact(str(out))


@contextmanager
def start_run(
    run_name: str | None = None,
    *,
    require_clean: bool = True,
    repo: Path = REPO_ROOT,
) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run stamped with lineage tags and the reproducibility artifacts.

    Tracking URI and credentials come from the environment (.env locally, secrets in CI).
    """
    load_dotenv(repo / ".env")
    if require_clean:
        ensure_clean_tree(repo)
    params = load_params(repo / "params.yaml")
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(lineage_tags(params, repo))
        mlflow.log_artifact(str(PARAMS_PATH if repo == REPO_ROOT else repo / "params.yaml"))
        _log_requirements_lock(repo)
        yield run
