"""Stamp every MLflow run with the lineage tags required by CLAUDE.md.

A registered model version must answer four questions from its tags alone:
which code, which data, which features, which pipeline version.
"""

import os
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

# mlflow prints a run-URL banner with a non-ASCII emoji on run end, which crashes on a Windows
# console using a non-UTF8 code page (cp1252). Suppress it; nothing here depends on that banner.
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")


class DirtyTreeError(RuntimeError):
    """Raised when training is attempted with uncommitted changes."""


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _git_lines(*args: str, cwd: Path) -> list[str]:
    """Like `_git`, but for output where per-line column positions matter (porcelain status): a
    whole-output `.strip()` would eat the leading space off just the first line and misalign it."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def git_commit(repo: Path = REPO_ROOT) -> str:
    return _git("rev-parse", "--short", "HEAD", cwd=repo)


def dvc_written_paths(repo: Path = REPO_ROOT) -> set[str]:
    """Git-tracked files that `dvc repro` itself writes: `dvc.lock` and every `cache: false`
    output, metric and plot declared in `dvc.yaml`. Cached outputs are git-ignored, so they never
    show up in `git status` and need no entry."""
    paths = {"dvc.lock"}
    dvc_yaml = repo / "dvc.yaml"
    if not dvc_yaml.exists():
        return paths
    stages = (yaml.safe_load(dvc_yaml.read_text(encoding="utf-8")) or {}).get("stages") or {}
    for stage in stages.values():
        for key in ("outs", "metrics", "plots"):
            for entry in stage.get(key) or []:
                if isinstance(entry, dict):
                    for path, opts in entry.items():
                        if isinstance(opts, dict) and opts.get("cache") is False:
                            paths.add(path)
    return paths


def ensure_clean_tree(repo: Path = REPO_ROOT, ignore: tuple[str, ...] = ()) -> None:
    """Refuse a dirty tree, except for changes to `ignore` (relative paths) and to the files
    `dvc repro` itself writes (see `dvc_written_paths`).

    `dvc repro` removes a stage's own declared outputs before running its command, which shows up as
    a pending change to that stage's own report file before the script gets a chance to write it.
    It also rewrites `dvc.lock` and the upstream stages' reports after each stage, so a multi-stage
    `dvc repro` would otherwise trip this check at `train` whenever an earlier stage had just run.
    None of that is the "uncommitted code/params" case this check exists for.
    """
    allowed = set(ignore) | dvc_written_paths(repo)
    lines = _git_lines("status", "--porcelain", cwd=repo)
    dirty = [line for line in lines if line[3:] not in allowed]
    if dirty:
        raise DirtyTreeError("Refusing to run: git tree has uncommitted changes.")


def dvc_data_md5(lock_path: Path = REPO_ROOT / "dvc.lock", data_path: str = DVC_DATA_PATH) -> str:
    """Return the md5 recorded in dvc.lock for `data_path` (default data/processed), or 'n/a'."""
    if not lock_path.exists():
        return NOT_AVAILABLE
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    for stage in (lock.get("stages") or {}).values():
        for out in stage.get("outs") or []:
            if out.get("path") == data_path:
                return str(out["md5"])
    return NOT_AVAILABLE


def split_spec(split: dict[str, str]) -> str:
    return (
        f"train {split['train_start']}..{split['train_end']}, "
        f"valid {split['valid_start']}..{split['valid_end']}, "
        f"test {split['test_start']}..{split['test_end']}"
    )


def lineage_tags(
    params: dict[str, Any],
    repo: Path = REPO_ROOT,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """The tags every run carries. A second dataset (Phase 7's ULB) passes `overrides` for the
    dataset, data hash, pipeline version and split spec; the tag names stay the same."""
    dataset = params["dataset"]
    tags = {
        "git_commit": git_commit(repo),
        "dataset_name": dataset["name"],
        "dataset_version": dataset["version"],
        "dvc_data_md5": dvc_data_md5(repo / "dvc.lock"),
        "pipeline_version": PIPELINE_VERSION,
        "split_spec": split_spec(params["split"]),
    }
    return {**tags, **(overrides or {})}


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
    ignore_dirty: tuple[str, ...] = (),
    experiment_name: str | None = None,
    lineage_overrides: dict[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run stamped with lineage tags and the reproducibility artifacts.

    Tracking URI and credentials come from the environment (.env locally, secrets in CI).
    `ignore_dirty` is forwarded to `ensure_clean_tree` (see its docstring). `experiment_name` and
    `lineage_overrides` let a secondary dataset log to its own experiment with its own tags.
    """
    load_dotenv(repo / ".env")
    if require_clean:
        ensure_clean_tree(repo, ignore=ignore_dirty)
    params = load_params(repo / "params.yaml")
    mlflow.set_experiment(experiment_name or params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(lineage_tags(params, repo, lineage_overrides))
        mlflow.log_artifact(str(PARAMS_PATH if repo == REPO_ROOT else repo / "params.yaml"))
        _log_requirements_lock(repo)
        yield run
