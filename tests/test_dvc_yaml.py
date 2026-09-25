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
