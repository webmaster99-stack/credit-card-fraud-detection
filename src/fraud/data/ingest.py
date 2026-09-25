"""Download the Sparkov simulated transactions from Kaggle into data/raw/sparkov.

Usage: uv run python -m fraud.data.ingest

Auth: a Kaggle access token in KAGGLE_API_TOKEN (or KAGGLE_ACCESS_TOKEN), read from .env.
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fraud.data.paths import RAW_DIR
from fraud.params import REPO_ROOT, load_params


def _authenticated_api() -> Any:
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Set KAGGLE_API_TOKEN (or KAGGLE_ACCESS_TOKEN) in .env.")
    os.environ["KAGGLE_API_TOKEN"] = token
    # Imported here on purpose: the kaggle package authenticates at import time.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def missing_files(directory: Path, expected: list[str]) -> list[str]:
    return [name for name in expected if not (directory / name).is_file()]


def ingest(dataset: str, expected: list[str], target: Path = RAW_DIR) -> None:
    target.mkdir(parents=True, exist_ok=True)
    api = _authenticated_api()
    api.dataset_download_files(dataset, path=str(target), unzip=True, quiet=False)
    missing = missing_files(target, expected)
    if missing:
        raise FileNotFoundError(f"Download is missing expected files: {missing}")


def main() -> None:
    cfg = load_params()["ingest"]
    ingest(cfg["kaggle_dataset"], cfg["expected_files"])


if __name__ == "__main__":
    main()
