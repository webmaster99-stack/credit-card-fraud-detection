"""Build the drift reference: a sample of the validation split plus the numbers alerts compare to.

Run by the `monitoring_reference` dvc stage. Never reads the test split's rows; the only
test-derived input is the recall already published in `reports/test_evaluation.json`.
"""

import json

import pandas as pd

from fraud.data.paths import PROCESSED_DIR
from fraud.params import REPO_ROOT, load_params
from fraud.serving.schema import REQUIRED_COLUMNS

REFERENCE_PATH = REPO_ROOT / "data" / "monitoring" / "reference.parquet"
STATS_PATH = REPO_ROOT / "reports" / "monitoring_reference.json"
TEST_EVALUATION_PATH = REPO_ROOT / "reports" / "test_evaluation.json"
TRAIN_METRICS_PATH = REPO_ROOT / "reports" / "train_metrics.json"


def main() -> None:
    params = load_params()
    cfg = params["monitoring"]
    valid = pd.read_parquet(PROCESSED_DIR / f"{cfg['reference_split']}.parquet")
    sample = valid.sample(
        n=min(int(cfg["reference_sample_rows"]), len(valid)), random_state=int(params["seed"])
    )[[*REQUIRED_COLUMNS, "card_id"]]
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.sort_values("trans_ts").to_parquet(REFERENCE_PATH, index=False)

    # Validation flag rate = flagged / rows = (recall * positives / precision) / rows.
    train = json.loads(TRAIN_METRICS_PATH.read_text(encoding="utf-8"))
    fraud_rate = float(valid["is_fraud"].mean())
    flag_rate = train["recall"] * fraud_rate / train["precision"]
    test = json.loads(TEST_EVALUATION_PATH.read_text(encoding="utf-8"))
    stats = {
        "reference_rows": len(sample),
        "validation_fraud_rate": fraud_rate,
        "validation_flag_rate": flag_rate,
        "test_recall": test["recall"],
        "test_precision": test["precision"],
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
