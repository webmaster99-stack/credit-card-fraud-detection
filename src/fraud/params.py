"""Loader for params.yaml, the single home of every tunable value."""

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# A non-editable install (the API image) puts this module in site-packages, where REPO_ROOT is
# meaningless, so the image points FRAUD_PARAMS_PATH at the params.yaml it ships.
PARAMS_PATH = Path(os.environ.get("FRAUD_PARAMS_PATH") or REPO_ROOT / "params.yaml")


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f)
    return params
