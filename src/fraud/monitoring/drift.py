"""Drift and alert logic for the nightly monitoring job (docs/plan.md Phase 6).

Everything here is a pure function over DataFrames, so the thresholds in `params.yaml`'s
`monitoring` section can be unit-tested without a database. Inputs are raw transaction columns
(the API's prediction log) plus three derived ones: `age`, `distance_km` and `hour`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

EARTH_RADIUS_KM = 6371.0
DAYS_PER_YEAR = 365.25
_EPS = 1e-4


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``df`` with `age` (years), `distance_km` (home to merchant) and `hour` added."""
    out = df.copy()
    ts = pd.to_datetime(out["trans_ts"])
    out["age"] = (ts - pd.to_datetime(out["dob"])).dt.days / DAYS_PER_YEAR
    out["hour"] = ts.dt.hour
    lat1, lon1 = np.radians(out["lat"]), np.radians(out["long"])
    lat2, lon2 = np.radians(out["merch_lat"]), np.radians(out["merch_long"])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    out["distance_km"] = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return out


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected / expected.sum(), _EPS, None)
    a = np.clip(actual / actual.sum(), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int) -> float:
    """Population stability index, with bin edges at the reference's quantiles."""
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant reference: compare the share equal to its single value
        edges = np.array([-np.inf, np.median(ref), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    return _psi(ref_counts, cur_counts)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    levels = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
    ref_counts = reference.astype(str).value_counts().reindex(levels, fill_value=0)
    cur_counts = current.astype(str).value_counts().reindex(levels, fill_value=0)
    return _psi(ref_counts.to_numpy(dtype=float), cur_counts.to_numpy(dtype=float))


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per monitored feature: its PSI, (numeric) Wasserstein distance and drift verdict."""
    ref, cur = add_derived(reference), add_derived(current)
    rows: list[dict[str, Any]] = []
    for col in cfg["numeric_features"]:
        psi = psi_numeric(ref[col], cur[col], int(cfg["psi_bins"]))
        scale = float(ref[col].std()) or 1.0
        wass = float(wasserstein_distance(ref[col].dropna(), cur[col].dropna())) / scale
        rows.append(_drift_row(col, "numeric", psi, cfg, wasserstein=wass))
    for col in cfg["categorical_features"]:
        rows.append(_drift_row(col, "categorical", psi_categorical(ref[col], cur[col]), cfg))
    return rows


def _drift_row(
    name: str, kind: str, psi: float, cfg: dict[str, Any], wasserstein: float | None = None
) -> dict[str, Any]:
    return {
        "feature": name,
        "kind": kind,
        "psi": round(psi, 4),
        "wasserstein_std": None if wasserstein is None else round(wasserstein, 4),
        "drifted": bool(psi > float(cfg["psi_alert"])),
    }


def flag_rate_status(
    flag_rate: float, reference_rate: float, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Flag rate against the validation flag rate, within [low, high] multiples."""
    ratio = flag_rate / reference_rate if reference_rate > 0 else float("inf")
    low, high = float(cfg["flag_rate_low"]), float(cfg["flag_rate_high"])
    return {
        "flag_rate": round(flag_rate, 6),
        "reference_flag_rate": round(reference_rate, 6),
        "ratio": None if not np.isfinite(ratio) else round(ratio, 3),
        "outside_band": bool(not (low <= ratio <= high)),
    }


def performance(labelled: pd.DataFrame, test_recall: float, cfg: dict[str, Any]) -> dict[str, Any]:
    """Recall/precision of flags on predictions whose true label has arrived."""
    n = len(labelled)
    if n < int(cfg["min_labels_for_performance"]):
        return {"n_labelled": n, "enough_labels": False}
    truth = labelled["true_label"].astype(bool)
    flagged = labelled["flagged"].astype(bool)
    tp = int((truth & flagged).sum())
    fn = int((truth & ~flagged).sum())
    fp = int((~truth & flagged).sum())
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    drop = None if recall is None else test_recall - recall
    return {
        "n_labelled": n,
        "enough_labels": True,
        "recall": recall,
        "precision": precision,
        "recall_drop": drop,
        "alert": bool(drop is not None and drop > float(cfg["recall_drop_points"])),
    }


def drift_streak(daily_drifted: list[bool]) -> int:
    """Consecutive drifted days ending at the most recent day."""
    streak = 0
    for drifted in reversed(daily_drifted):
        if not drifted:
            break
        streak += 1
    return streak


def build_alerts(
    *,
    daily: list[dict[str, Any]],
    perf: dict[str, Any],
    invalid_fraction: float,
    error_rate: float | None,
    cfg: dict[str, Any],
) -> list[dict[str, str]]:
    """Alerts per the plan's table. ``daily``: oldest first, each with `n`, `n_drifted`, `flag`."""
    alerts: list[dict[str, str]] = []
    if error_rate is not None and error_rate > float(cfg["max_error_rate"]):
        alerts.append({"layer": "service", "message": f"Error rate {error_rate:.1%} is above 1%."})
    if invalid_fraction > float(cfg["max_invalid_fraction"]):
        alerts.append(
            {
                "layer": "data_quality",
                "message": f"{invalid_fraction:.2%} of requests were invalid.",
            }
        )
    usable = [d for d in daily if d["n"] >= int(cfg["min_rows_per_day"])]
    days = int(cfg["drift_days"])
    recent = usable[-days:]
    if len(recent) == days and all(
        d["n_drifted"] >= int(cfg["drift_features_min"]) for d in recent
    ):
        alerts.append(
            {"layer": "data_drift", "message": f"Feature drift on {days} consecutive days."}
        )
    if usable and usable[-1]["flag"]["outside_band"]:
        alerts.append(
            {"layer": "prediction_drift", "message": "Flag rate is outside the expected band."}
        )
    if perf.get("alert"):
        points = perf["recall_drop"] * 100
        alerts.append(
            {"layer": "performance", "message": f"Recall dropped {points:.0f} points vs test."}
        )
    return alerts
