import subprocess
from pathlib import Path

import pytest

from fraud.features import PIPELINE_VERSION
from fraud.models.lineage import (
    NOT_AVAILABLE,
    DirtyTreeError,
    dvc_data_md5,
    ensure_clean_tree,
    git_commit,
    lineage_tags,
    split_spec,
)
from fraud.params import load_params


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("a")
    git("add", ".")
    git("commit", "-m", "init")
    return tmp_path


def test_clean_tree_passes(repo: Path) -> None:
    ensure_clean_tree(repo)


def test_dirty_tree_refused(repo: Path) -> None:
    (repo / "a.txt").write_text("changed")
    with pytest.raises(DirtyTreeError):
        ensure_clean_tree(repo)


def test_dirty_tree_ignores_named_paths(repo: Path) -> None:
    # Simulates dvc repro removing a stage's own declared output before the stage's command runs.
    (repo / "a.txt").unlink()
    with pytest.raises(DirtyTreeError):
        ensure_clean_tree(repo)
    ensure_clean_tree(repo, ignore=("a.txt",))


def test_dirty_tree_ignore_does_not_hide_other_changes(repo: Path) -> None:
    (repo / "a.txt").unlink()
    (repo / "b.txt").write_text("new")
    with pytest.raises(DirtyTreeError):
        ensure_clean_tree(repo, ignore=("a.txt",))


def test_dirty_tree_ignores_dvc_written_files(repo: Path) -> None:
    # A multi-stage `dvc repro` rewrites dvc.lock and upstream cache:false reports before `train`.
    (repo / "dvc.yaml").write_text(
        "stages:\n  s:\n    cmd: x\n    outs:\n      - data/raw\n"
        "    metrics:\n      - reports/m.json:\n          cache: false\n"
    )
    (repo / "reports").mkdir()
    (repo / "reports" / "m.json").write_text("{}")
    (repo / "dvc.lock").write_text("a")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "dvc"], cwd=repo, check=True, capture_output=True)
    (repo / "reports" / "m.json").write_text('{"x": 1}')
    (repo / "dvc.lock").write_text("b")
    ensure_clean_tree(repo)
    (repo / "a.txt").write_text("changed")
    with pytest.raises(DirtyTreeError):
        ensure_clean_tree(repo)


def test_git_commit_is_short_sha(repo: Path) -> None:
    assert len(git_commit(repo)) >= 7


def test_dvc_md5_missing_lock(tmp_path: Path) -> None:
    assert dvc_data_md5(tmp_path / "dvc.lock") == NOT_AVAILABLE


def test_dvc_md5_read_from_lock(tmp_path: Path) -> None:
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "schema: '2.0'\nstages:\n  split:\n    outs:\n"
        "    - path: data/processed\n      md5: abc123.dir\n"
    )
    assert dvc_data_md5(lock) == "abc123.dir"


def test_split_spec_format() -> None:
    spec = split_spec(load_params()["split"])
    assert spec.startswith("train 2019-01-01..2020-06-30")


def test_lineage_tags_complete(repo: Path) -> None:
    tags = lineage_tags(load_params(), repo)
    assert set(tags) == {
        "git_commit",
        "dataset_name",
        "dataset_version",
        "dvc_data_md5",
        "pipeline_version",
        "split_spec",
    }
    assert tags["pipeline_version"] == PIPELINE_VERSION
