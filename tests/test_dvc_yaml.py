import pytest
import yaml

from fraud.data.paths import INTERIM_PATH, PROCESSED_DIR, RAW_DIR
from fraud.params import REPO_ROOT


def _rel(path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def test_dvc_outs_match_path_constants() -> None:
    stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    assert stages["ingest"]["outs"] == [_rel(RAW_DIR)]
    assert stages["clean"]["outs"] == [_rel(INTERIM_PATH)]
    assert stages["split"]["outs"] == [_rel(PROCESSED_DIR)]
    assert _rel(RAW_DIR) in stages["clean"]["deps"]
    assert _rel(INTERIM_PATH) in stages["split"]["deps"]


@pytest.mark.parametrize("stage_name", ["features", "train", "evaluate"])
def test_stage_never_depends_on_the_test_split(stage_name: str) -> None:
    stage = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"][
        stage_name
    ]
    deps = stage["deps"]
    assert f"{_rel(PROCESSED_DIR)}/train.parquet" in deps
    assert f"{_rel(PROCESSED_DIR)}/valid.parquet" in deps
    assert not any("test" in d and "parquet" in d for d in deps)
    assert not any(d == _rel(PROCESSED_DIR) for d in deps)
