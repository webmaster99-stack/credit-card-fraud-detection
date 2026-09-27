"""Stateful (v2) card-history features: velocity and behavioural groups.

Definition, shared by training, batch scoring and (Phase 5) the online API: for a transaction at
time ``t`` on card ``c``, the *history* is every other transaction on ``c`` with a timestamp
**strictly earlier** than ``t``. The transaction itself, later transactions and transactions with
the same timestamp are never used. That makes each value independent of row order and of anything
that happens after ``t``, so it cannot leak the future (tests/test_features_history.py).

Velocity windows are ``[t - w, t)``. The online store must reproduce exactly these definitions.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from fraud.features.transformers import FrameTransformer, require_columns

NS_PER_HOUR = 3_600_000_000_000
NS_PER_DAY = 24 * NS_PER_HOUR

FloatArray = npt.NDArray[np.float64]


def velocity_column_names(windows_hours: Sequence[int]) -> list[str]:
    counts = [f"txn_count_{w}h" for w in windows_hours]
    spends = [f"log_spend_{w}h" for w in windows_hours]
    return [*counts, *spends]


BEHAVIOURAL_COLUMNS = (
    "amt_vs_card_mean",
    "hours_since_last_txn",
    "is_first_category_use",
    "is_first_card_txn",
)


def _group_segments(
    codes: npt.NDArray[np.int64], ts: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.intp], list[tuple[int, int]]]:
    """Sort rows by (group, time) and return the order plus each group's [start, end) slice."""
    order = np.lexsort((ts, codes))
    sorted_codes = codes[order]
    cuts = np.flatnonzero(np.diff(sorted_codes)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(order)]))
    return order, [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _strictly_earlier_counts(codes: npt.NDArray[np.int64], ts: npt.NDArray[np.int64]) -> FloatArray:
    """For each row, how many rows of the same group have a strictly earlier timestamp."""
    order, segments = _group_segments(codes, ts)
    t_sorted = ts[order]
    out = np.zeros(len(ts))
    for s, e in segments:
        t = t_sorted[s:e]
        out[order[s:e]] = np.searchsorted(t, t, side="left")
    return out


def card_history_features(df: pd.DataFrame, windows_hours: Sequence[int]) -> pd.DataFrame:
    """Velocity and behavioural features for every row, from earlier rows of the same card."""
    require_columns(df, ("trans_ts", "card_id", "amt", "category"), "card_history_features")
    n = len(df)
    ts = df["trans_ts"].astype("datetime64[ns]").to_numpy().astype(np.int64)
    amt = df["amt"].to_numpy(float)
    card_codes = pd.factorize(df["card_id"].astype(str))[0].astype(np.int64)
    pair_codes = pd.factorize(df["card_id"].astype(str) + "|" + df["category"].astype(str))[
        0
    ].astype(np.int64)

    counts = {w: np.zeros(n) for w in windows_hours}
    spends = {w: np.zeros(n) for w in windows_hours}
    prior_n = np.zeros(n)
    prior_sum = np.zeros(n)
    since_last = np.full(n, np.nan)

    order, segments = _group_segments(card_codes, ts)
    t_sorted, a_sorted = ts[order], amt[order]
    for s, e in segments:
        t, a = t_sorted[s:e], a_sorted[s:e]
        cum = np.concatenate(([0.0], np.cumsum(a)))
        hi = np.searchsorted(t, t, side="left")  # rows strictly before t
        idx = order[s:e]
        prior_n[idx] = hi
        prior_sum[idx] = cum[hi]
        has_prior = hi > 0
        last = np.where(has_prior, t[np.maximum(hi - 1, 0)], 0)
        since_last[idx] = np.where(has_prior, (t - last) / NS_PER_HOUR, np.nan)
        for w in windows_hours:
            lo = np.searchsorted(t, t - w * NS_PER_HOUR, side="left")  # first row with ts >= t - w
            counts[w][idx] = hi - lo
            # Cumulative-sum differences can leave tiny negative float noise; spend is never < 0.
            spends[w][idx] = np.maximum(cum[hi] - cum[lo], 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prior_n > 0, amt / (prior_sum / np.maximum(prior_n, 1)), np.nan)

    columns: dict[str, FloatArray] = {f"txn_count_{w}h": counts[w] for w in windows_hours}
    columns.update({f"log_spend_{w}h": np.log1p(spends[w]) for w in windows_hours})
    columns["amt_vs_card_mean"] = ratio
    columns["hours_since_last_txn"] = since_last
    columns["is_first_category_use"] = (_strictly_earlier_counts(pair_codes, ts) == 0).astype(float)
    columns["is_first_card_txn"] = (prior_n == 0).astype(float)
    return pd.DataFrame(columns, index=df.index)


class CardHistoryFeatures(FrameTransformer):
    """Velocity and behavioural features from the card's earlier transactions.

    History is whatever rows are in the frame being transformed, so a batch scores against itself.
    To score against earlier stored rows, use ``fraud.features.pipeline.transform_with_context``.
    Rows may arrive in any order; output keeps the input order.
    """

    required = ("trans_ts", "card_id", "amt", "category")

    def __init__(self, windows_hours: Sequence[int] = (1, 24, 168)) -> None:
        self.windows_hours = windows_hours

    def get_feature_names_out(self, input_features: object = None) -> npt.NDArray[np.object_]:
        names = [*velocity_column_names(self.windows_hours), *BEHAVIOURAL_COLUMNS]
        return np.asarray(names, dtype=object)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return card_history_features(X, self.windows_hours)
