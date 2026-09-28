from importlib.metadata import version
from pathlib import Path

import pytest
from build_space import SPACE_FILES, stage_space

from fraud.params import REPO_ROOT


def _pins() -> dict[str, str]:
    lines = (REPO_ROOT / "demo" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    specs = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return dict(spec.split("==") for spec in specs)


@pytest.mark.parametrize(("package", "pinned"), sorted(_pins().items()))
def test_space_pins_match_the_locked_environment(package: str, pinned: str) -> None:
    """The Space must unpickle and score with the library versions the bundle was built under."""
    assert version(package) == pinned


def test_staged_space_has_app_params_and_the_vendored_package(tmp_path: Path) -> None:
    staged = stage_space(tmp_path / "space")

    for name in [*SPACE_FILES, "params.yaml", "src/fraud/serving/model.py"]:
        assert (staged / name).is_file(), name
    assert not list(staged.rglob("__pycache__"))
    # the vendored layout keeps `fraud.params.REPO_ROOT` pointing at the Space root
    assert (staged / "src" / "fraud" / "params.py").parents[2] == staged
    assert (staged / "README.md").read_text(encoding="utf-8").startswith("---\n")
