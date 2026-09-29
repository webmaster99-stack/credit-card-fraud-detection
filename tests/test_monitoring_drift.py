import numpy as np
import pandas as pd
import pytest

from fraud.monitoring.drift import (
    add_derived,
    build_alerts,
    drift_streak,
    feature_drift,
    flag_rate_status,
    performance,
    psi_categorical,
    psi_numeric,
)
from fraud.params import load_params

CFG = load_params()["monitoring"]


def _txns(n: int, seed: int, amt_scale: float = 1.0, category: str = "grocery_pos") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "trans_ts": pd.Timestamp("2020-08-01")
            + pd.to_timedelta(rng.integers(0, 86400, n), "s"),
            "amt": rng.lognormal(3.5, 1.0, n) * amt_scale,
            "category": category,
            "gender": rng.choice(["F", "M"], n),
            "state": rng.choice(["IL", "TX", "NY"], n),
            "city_pop": rng.integers(1_000, 500_000, n),
            "dob": pd.Timestamp("1980-01-01"),
            "lat": 40.0,
            "long": -89.0,
            "merch_lat": rng.uniform(39, 41, n),
            "merch_long": rng.uniform(-90, -88, n),
        }
    )


def test_psi_is_near_zero_for_same_distribution_and_large_for_shift() -> None:
    a, b = _txns(5000, 1)["amt"], _txns(5000, 2)["amt"]
    assert psi_numeric(a, b, 10) < 0.05
    assert psi_numeric(a, b * 5, 10) > 0.25


def test_psi_categorical_detects_new_mix() -> None:
    ref = pd.Series(["a"] * 500 + ["b"] * 500)
    assert psi_categorical(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi_categorical(ref, pd.Series(["a"] * 950 + ["c"] * 50)) > 0.25


def test_add_derived_columns() -> None:
    out = add_derived(_txns(10, 3))
    assert out["age"].between(39, 41).all()
    assert out["hour"].between(0, 23).all()
    assert (out["distance_km"] >= 0).all()


def test_feature_drift_flags_only_the_shifted_feature() -> None:
    ref = _txns(4000, 1)
    rows = {r["feature"]: r for r in feature_drift(ref, _txns(1000, 2, amt_scale=4.0), CFG)}
    assert rows["amt"]["drifted"]
    assert not rows["city_pop"]["drifted"]
    assert not rows["gender"]["drifted"]


def test_flag_rate_band() -> None:
    assert not flag_rate_status(0.01, 0.01, CFG)["outside_band"]
    assert flag_rate_status(0.05, 0.01, CFG)["outside_band"]
    assert flag_rate_status(0.001, 0.01, CFG)["outside_band"]


def test_performance_needs_enough_labels_then_alerts_on_recall_drop() -> None:
    few = pd.DataFrame({"true_label": [True] * 3, "flagged": [True] * 3})
    assert performance(few, 0.98, CFG)["enough_labels"] is False
    n = int(CFG["min_labels_for_performance"])
    labelled = pd.DataFrame(
        {"true_label": [True] * n, "flagged": [True] * (n // 2) + [False] * (n - n // 2)}
    )
    result = performance(labelled, 0.98, CFG)
    assert result["recall"] == pytest.approx((n // 2) / n)
    assert result["alert"] is True


def test_drift_streak() -> None:
    assert drift_streak([False, True, True]) == 2
    assert drift_streak([True, False]) == 0


def _day(n: int, drifted: int, outside: bool = False) -> dict:
    return {"n": n, "n_drifted": drifted, "flag": {"outside_band": outside}}


def test_alerts_need_three_consecutive_drift_days() -> None:
    quiet = {"alert": False}
    kw = {"perf": quiet, "invalid_fraction": 0.0, "error_rate": 0.0, "cfg": CFG}
    n = int(CFG["min_rows_per_day"])
    two = build_alerts(daily=[_day(n, 0), _day(n, 2), _day(n, 2)], **kw)
    three = build_alerts(daily=[_day(n, 2), _day(n, 2), _day(n, 2)], **kw)
    assert not [a for a in two if a["layer"] == "data_drift"]
    assert [a for a in three if a["layer"] == "data_drift"]


def test_alerts_for_service_quality_and_flag_rate() -> None:
    n = int(CFG["min_rows_per_day"])
    alerts = build_alerts(
        daily=[_day(n, 0, outside=True)],
        perf={"alert": False},
        invalid_fraction=0.02,
        error_rate=0.05,
        cfg=CFG,
    )
    assert {a["layer"] for a in alerts} == {"service", "data_quality", "prediction_drift"}


def test_thin_days_never_alert() -> None:
    assert (
        build_alerts(
            daily=[_day(3, 5, outside=True)] * 3,
            perf={},
            invalid_fraction=0.0,
            error_rate=None,
            cfg=CFG,
        )
        == []
    )
