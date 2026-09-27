"""Leakage and correctness tests for the card-history features (v2)."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_clean_frame
from fraud.features.history import CardHistoryFeatures, card_history_features

WINDOWS = (1, 24, 168)


def _txns(rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
    """rows: (timestamp, card, amount, category)."""
    return pd.DataFrame(
        {
            "trans_ts": pd.to_datetime([r[0] for r in rows]),
            "card_id": [r[1] for r in rows],
            "amt": [r[2] for r in rows],
            "category": [r[3] for r in rows],
        }
    )


def test_hand_computed_values() -> None:
    df = _txns(
        [
            ("2020-01-01 00:00:00", "a", 10.0, "home"),
            ("2020-01-01 00:30:00", "a", 20.0, "home"),
            ("2020-01-01 02:00:00", "a", 60.0, "travel"),
            ("2020-01-01 01:00:00", "b", 5.0, "home"),  # another card must not leak in
        ]
    ).sort_values("trans_ts", kind="stable", ignore_index=True)
    out = card_history_features(df, WINDOWS)
    a = out[df["card_id"] == "a"].reset_index(drop=True)

    assert a["txn_count_1h"].tolist() == [0, 1, 0]  # 3rd txn: 00:30 is 90 min before 02:00
    assert a["txn_count_24h"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(a["log_spend_24h"], np.log1p([0.0, 10.0, 30.0]))
    assert np.isnan(a["amt_vs_card_mean"].iloc[0])
    assert a["amt_vs_card_mean"].iloc[1] == pytest.approx(20.0 / 10.0)
    assert a["amt_vs_card_mean"].iloc[2] == pytest.approx(60.0 / 15.0)
    assert np.isnan(a["hours_since_last_txn"].iloc[0])
    assert a["hours_since_last_txn"].iloc[1:].tolist() == pytest.approx([0.5, 1.5])
    assert a["is_first_category_use"].tolist() == [1, 0, 1]
    assert a["is_first_card_txn"].tolist() == [1, 0, 0]


def test_window_is_half_open_and_excludes_current_row() -> None:
    df = _txns(
        [
            ("2020-01-01 00:00:00", "a", 10.0, "home"),
            ("2020-01-01 01:00:00", "a", 10.0, "home"),  # exactly 1 h later: inside [t-1h, t)
            ("2020-01-01 02:00:01", "a", 10.0, "home"),  # 1 h 1 s after the previous: just outside
        ]
    )
    out = card_history_features(df, WINDOWS)
    assert out["txn_count_1h"].tolist() == [0, 1, 0]


def test_same_timestamp_rows_do_not_see_each_other() -> None:
    df = _txns(
        [
            ("2020-01-01 00:00:00", "a", 10.0, "home"),
            ("2020-01-01 00:00:00", "a", 99.0, "home"),
        ]
    )
    out = card_history_features(df, WINDOWS)
    assert out["txn_count_24h"].tolist() == [0, 0]
    assert out["is_first_card_txn"].tolist() == [1, 1]


def test_future_rows_do_not_change_earlier_features() -> None:
    """Leakage test: features at time t are identical whether or not later rows exist."""
    df = make_clean_frame(n=400)
    cut = df["trans_ts"].iloc[200]
    full = card_history_features(df, WINDOWS)
    early_rows = df["trans_ts"] <= cut
    truncated = card_history_features(df[early_rows], WINDOWS)
    pd.testing.assert_frame_equal(full[early_rows], truncated)


def test_perturbing_the_future_changes_nothing_earlier() -> None:
    df = make_clean_frame(n=400)
    cut = df["trans_ts"].iloc[200]
    tampered = df.copy()
    later = tampered["trans_ts"] > cut
    tampered.loc[later, "amt"] *= 1000
    tampered.loc[later, "category"] = "travel"
    a = card_history_features(df, WINDOWS)
    b = card_history_features(tampered, WINDOWS)
    pd.testing.assert_frame_equal(a[~later], b[~later])


def test_own_amount_and_label_never_feed_own_features() -> None:
    df = make_clean_frame(n=200)
    tampered = df.copy()
    tampered["amt"] = tampered["amt"] * 7
    base = card_history_features(df, WINDOWS)
    other = card_history_features(tampered, WINDOWS)
    # Counts and timing depend only on timestamps, so scaling every amount leaves them unchanged.
    for col in ["txn_count_1h", "txn_count_24h", "txn_count_168h", "hours_since_last_txn"]:
        pd.testing.assert_series_equal(base[col], other[col])
    # amt / prior mean is scale-free; it would change if the row's own amount leaked into the mean.
    pd.testing.assert_series_equal(base["amt_vs_card_mean"], other["amt_vs_card_mean"])


def test_row_order_does_not_matter() -> None:
    df = make_clean_frame(n=300)
    shuffled = df.sample(frac=1.0, random_state=0)
    a = card_history_features(df, WINDOWS)
    b = card_history_features(shuffled, WINDOWS).loc[df.index]
    pd.testing.assert_frame_equal(a, b)


def test_transformer_matches_function_and_declares_columns(clean_frame: pd.DataFrame) -> None:
    t = CardHistoryFeatures(WINDOWS)
    out = t.transform(clean_frame)
    assert list(out.columns) == list(t.get_feature_names_out())
    pd.testing.assert_frame_equal(out, card_history_features(clean_frame, WINDOWS))


def test_windows_come_from_arguments() -> None:
    out = CardHistoryFeatures((2, 5)).transform(make_clean_frame(n=50))
    assert {"txn_count_2h", "txn_count_5h", "log_spend_2h", "log_spend_5h"} <= set(out.columns)
    assert "txn_count_24h" not in out.columns
