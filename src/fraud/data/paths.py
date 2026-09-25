"""Fixed locations of pipeline data. Keep in sync with dvc.yaml (tests/test_dvc_yaml.py)."""

from fraud.params import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw" / "sparkov"
INTERIM_PATH = REPO_ROOT / "data" / "interim" / "transactions.parquet"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLIT_NAMES = ("train", "valid", "test")
