"""Time-based train/valid/test split. Never random: the model scores the future in production.

Usage: uv run python -m fraud.data.split
"""

import json
from typing import Any

import pandas as pd

from fraud.data.paths import INTERIM_PATH, PROCESSED_DIR, SPLIT_NAMES
from fraud.params import REPO_ROOT, load_params

SUMMARY_PATH = REPO_ROOT / "reports" / "split_summary.json"


def windows(split: dict[str, str]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Half-open [start, end + 1 day) windows per split, validated to be ordered and disjoint."""
    out = {
        name: (pd.Timestamp(split[f"{name}_start"]), pd.Timestamp(split[f"{name}_end"]))
        for name in SPLIT_NAMES
    }
    out = {name: (start, end + pd.Timedelta(days=1)) for name, (start, end) in out.items()}
    ordered = [out[name] for name in SPLIT_NAMES]
    for (start, end), (next_start, _) in zip(ordered, ordered[1:], strict=False):
        if not start < end <= next_start:
            raise ValueError("Split windows must be ordered and non-overlapping.")
    if not ordered[-1][0] < ordered[-1][1]:
        raise ValueError("Split windows must be ordered and non-overlapping.")
    return out


def assign_split(ts: pd.Series, split: dict[str, str]) -> pd.Series:
    """Label every row train/valid/test by timestamp; fail if any row is in no window."""
    labels = pd.Series(pd.NA, index=ts.index, dtype="object")
    for name, (start, end) in windows(split).items():
        labels[(ts >= start) & (ts < end)] = name
    unassigned = labels.isna()
    if unassigned.any():
        raise ValueError(
            f"{int(unassigned.sum())} rows fall outside all split windows "
            f"({ts[unassigned].min()} .. {ts[unassigned].max()})."
        )
    return labels


def summarize(parts: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        name: {
            "rows": len(part),
            "frauds": int(part["is_fraud"].sum()),
            "fraud_rate": float(part["is_fraud"].mean()),
            "start": str(part["trans_ts"].min()),
            "end": str(part["trans_ts"].max()),
            "cards": int(part["card_id"].nunique()),
        }
        for name, part in parts.items()
    }


def split_frame(df: pd.DataFrame, split: dict[str, str]) -> dict[str, pd.DataFrame]:
    labels = assign_split(df["trans_ts"], split)
    return {name: df[labels == name].reset_index(drop=True) for name in SPLIT_NAMES}


def main() -> None:
    parts = split_frame(pd.read_parquet(INTERIM_PATH), load_params()["split"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    summary = summarize(parts)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, s in summary.items():
        print(f"{name}: {s['rows']:,} rows, {s['frauds']:,} frauds ({s['fraud_rate']:.3%})")


if __name__ == "__main__":
    main()
