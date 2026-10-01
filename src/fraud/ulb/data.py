"""ULB ingest and time-based split (dvc.yaml `ulb_ingest`, `ulb_split`).

ULB's only time information is `Time`, seconds since the first transaction (about 48 hours), so
the split is by Time: earlier hours train, then validation, the latest hours test. Never random.

Usage: uv run python -m fraud.ulb.data ingest|split
"""

import argparse
import json
from typing import Any

import pandas as pd
import pandera.pandas as pa

from fraud.data.ingest import ingest
from fraud.params import load_params
from fraud.ulb.paths import (
    PROCESSED_DIR,
    RAW_DIR,
    RAW_FILE,
    SPLIT_NAMES,
    SPLIT_SUMMARY_PATH,
)

SECONDS_PER_HOUR = 3600
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
TARGET = "Class"

SCHEMA = pa.DataFrameSchema(
    {
        "Time": pa.Column(float, pa.Check.ge(0)),
        **{c: pa.Column(float) for c in PCA_COLUMNS},
        "Amount": pa.Column(float, pa.Check.ge(0)),
        TARGET: pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
)


def split_spec_text(split: dict[str, int]) -> str:
    return (
        f"train Time<{split['train_end_hour']}h, "
        f"valid {split['train_end_hour']}h..{split['valid_end_hour']}h, "
        f"test Time>={split['valid_end_hour']}h"
    )


def assign_split(time: pd.Series, split: dict[str, int]) -> pd.Series:
    """Label rows train/valid/test by Time. Windows are [0, train_end), [train_end, valid_end),
    [valid_end, inf) in hours; ordering is validated."""
    train_end, valid_end = split["train_end_hour"], split["valid_end_hour"]
    if not 0 < train_end < valid_end:
        raise ValueError("Split hours must satisfy 0 < train_end_hour < valid_end_hour.")
    hours = time / SECONDS_PER_HOUR
    labels = pd.Series("test", index=time.index, dtype="object")
    labels[hours < valid_end] = "valid"
    labels[hours < train_end] = "train"
    return labels


def split_frame(df: pd.DataFrame, split: dict[str, int]) -> dict[str, pd.DataFrame]:
    labels = assign_split(df["Time"], split)
    return {name: df[labels == name].reset_index(drop=True) for name in SPLIT_NAMES}


def summarize(parts: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        name: {
            "rows": len(part),
            "frauds": int(part[TARGET].sum()),
            "fraud_rate": float(part[TARGET].mean()),
            "start_hour": float(part["Time"].min() / SECONDS_PER_HOUR),
            "end_hour": float(part["Time"].max() / SECONDS_PER_HOUR),
        }
        for name, part in parts.items()
    }


def load_raw() -> pd.DataFrame:
    df = SCHEMA.validate(pd.read_csv(RAW_FILE))
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["ingest", "split"])
    command = parser.parse_args().command
    cfg = load_params()["ulb"]
    if command == "ingest":
        ingest(cfg["kaggle_dataset"], cfg["expected_files"], target=RAW_DIR)
        return
    parts = split_frame(load_raw(), cfg["split"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    summary = summarize(parts)
    SPLIT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for name, s in summary.items():
        print(f"{name}: {s['rows']:,} rows, {s['frauds']:,} frauds ({s['fraud_rate']:.3%})")


if __name__ == "__main__":
    main()
