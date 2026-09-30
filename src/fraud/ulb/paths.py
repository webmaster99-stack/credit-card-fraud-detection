"""Fixed locations of the ULB data. Keep in sync with dvc.yaml (tests/test_dvc_yaml.py)."""

from fraud.params import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw" / "ulb"
RAW_FILE = RAW_DIR / "creditcard.csv"
# A sibling of data/processed, not a child: DVC forbids nested outputs.
PROCESSED_DIR = REPO_ROOT / "data" / "processed_ulb"
SPLIT_NAMES = ("train", "valid", "test")
MODEL_PATH = REPO_ROOT / "data" / "models" / "ulb_pipeline.joblib"
REPORTS_DIR = REPO_ROOT / "reports"
SPLIT_SUMMARY_PATH = REPORTS_DIR / "ulb_split_summary.json"
LADDER_PATH = REPORTS_DIR / "ulb_ladder.json"
TEST_EVALUATION_PATH = REPORTS_DIR / "ulb_test_evaluation.json"
