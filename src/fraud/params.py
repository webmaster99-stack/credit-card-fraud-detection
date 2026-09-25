"""Loader for params.yaml, the single home of every tunable value."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f)
    return params
