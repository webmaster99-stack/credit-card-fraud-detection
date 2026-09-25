import pandas as pd
import pytest

from fraud.data.split import assign_split, split_frame, summarize, windows
from fraud.params import load_params

SPLIT = load_params()["split"]


def _frame(stamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trans_ts": pd.to_datetime(stamps),
            "card_id": ["c"] * len(stamps),
            "is_fraud": [0] * len(stamps),
        }
    )


def test_boundaries_are_inclusive_of_whole_end_day() -> None:
    ts = pd.Series(
        pd.to_datetime(
            [
                "2019-01-01 00:00:00",
                "2020-06-30 23:59:59",
                "2020-07-01 00:00:00",
                "2020-09-30 23:59:59",
                "2020-10-01 00:00:00",
                "2020-12-31 23:59:59",
            ]
        )
    )
    assert assign_split(ts, SPLIT).tolist() == ["train", "train", "valid", "valid", "test", "test"]


def test_rows_outside_all_windows_raise() -> None:
    with pytest.raises(ValueError, match="outside all split windows"):
        assign_split(pd.Series(pd.to_datetime(["2018-12-31 23:59:59"])), SPLIT)


def test_overlapping_windows_rejected() -> None:
    bad = {**SPLIT, "valid_start": "2020-06-01"}
    with pytest.raises(ValueError, match="non-overlapping"):
        windows(bad)


def test_split_is_disjoint_complete_and_ordered() -> None:
    df = _frame(["2019-03-01", "2020-01-01", "2020-07-15", "2020-08-01", "2020-11-11"])
    parts = split_frame(df, SPLIT)
    assert sum(len(p) for p in parts.values()) == len(df)
    assert parts["train"]["trans_ts"].max() < parts["valid"]["trans_ts"].min()
    assert parts["valid"]["trans_ts"].max() < parts["test"]["trans_ts"].min()


def test_summary_reports_fraud_rate() -> None:
    part = _frame(["2019-03-01", "2019-03-02"])
    part.loc[0, "is_fraud"] = 1
    s = summarize({"train": part})["train"]
    assert (s["rows"], s["frauds"], s["fraud_rate"]) == (2, 1, 0.5)
