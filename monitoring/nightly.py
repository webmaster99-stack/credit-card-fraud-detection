"""Nightly monitoring job: read the prediction log, run every layer's checks, store the summary.

    uv run --group monitoring python -m monitoring.nightly

Needs DATABASE_URL (the API's Neon database) and the drift reference from `dvc pull`
(`data/monitoring/reference.parquet`, `reports/monitoring_reference.json`). The summary lands in
`monitoring_reports` (kind `nightly`), where the API serves it to the web monitoring page; the
Evidently HTML report is stored beside it. Exits 1 when any alert fires, so the scheduled GitHub
Action turns red and emails the owner.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from api.db import close_pool, init_schema, open_pool, save_monitoring_report
from dotenv import load_dotenv
from psycopg.rows import dict_row

from fraud.monitoring.drift import (
    build_alerts,
    feature_drift,
    flag_rate_status,
    performance,
)
from fraud.params import REPO_ROOT, load_params

REFERENCE_PATH = REPO_ROOT / "data" / "monitoring" / "reference.parquet"
STATS_PATH = REPO_ROOT / "reports" / "monitoring_reference.json"

_TXN_COLUMNS = [
    "trans_ts", "amt", "category", "gender", "state", "city_pop", "dob",
    "lat", "long", "merch_lat", "merch_long",
]  # fmt: skip


def read_predictions(pool: Any, days: int, source: str | None = None) -> pd.DataFrame:
    """Logged predictions from the last ``days`` days, with a `day` column (creation date, UTC)."""
    columns = ", ".join([*_TXN_COLUMNS, "flagged", "fraud_probability", "true_label", "created_at"])
    query = (
        f"SELECT {columns} FROM predictions "  # noqa: S608
        "WHERE created_at >= now() - make_interval(days => %(days)s)"
    )
    params: dict[str, Any] = {"days": days}
    if source is not None:
        query += " AND source = %(source)s"
        params["source"] = source
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(query, params).fetchall()
    frame = pd.DataFrame(
        rows, columns=[*_TXN_COLUMNS, "flagged", "fraud_probability", "true_label", "created_at"]
    )
    frame["day"] = pd.to_datetime(frame["created_at"], utc=True).dt.strftime("%Y-%m-%d")
    return frame


def read_request_stats(pool: Any, days: int) -> dict[str, Any]:
    """Error rate (5xx) and invalid-input rate (422) over `/v1/predict*` requests."""
    query = (
        "SELECT count(*) AS n, "
        "count(*) FILTER (WHERE status_code >= 500) AS errors, "
        "count(*) FILTER (WHERE status_code = 422) AS invalid, "
        "percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms "
        "FROM api_requests WHERE path LIKE '/v1/predict%%' "
        "AND created_at >= now() - make_interval(days => %(days)s)"
    )
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(query, {"days": days}).fetchone()
    assert row is not None
    n = int(row["n"])
    return {
        "requests": n,
        "error_rate": row["errors"] / n if n else None,
        "invalid_fraction": row["invalid"] / n if n else 0.0,
        "p95_latency_ms": None if row["p95_ms"] is None else round(float(row["p95_ms"]), 1),
    }


def evidently_html(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> str | None:
    """Evidently's data-drift report as an HTML string; None if Evidently is not installed."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return None
    from fraud.monitoring.drift import add_derived

    ref, cur = add_derived(reference)[columns], add_derived(current)[columns]
    snapshot = Report([DataDriftPreset()]).run(cur, ref)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.html"
        snapshot.save_html(str(path))
        return path.read_text(encoding="utf-8")


def build_summary(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    stats: dict[str, Any],
    requests: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    daily: list[dict[str, Any]] = []
    for day, group in predictions.groupby("day"):
        n = len(group)
        entry: dict[str, Any] = {"day": day, "n": n}
        if n >= int(cfg["min_rows_per_day"]):
            drift = feature_drift(reference, group, cfg)
            entry["features"] = drift
            entry["n_drifted"] = sum(f["drifted"] for f in drift)
            entry["flag"] = flag_rate_status(
                float(group["flagged"].mean()), stats["validation_flag_rate"], cfg
            )
        else:
            entry["n_drifted"] = 0
            entry["flag"] = {"outside_band": False}
        daily.append(entry)
    daily.sort(key=lambda d: d["day"])

    labelled = predictions[predictions["true_label"].notna()]
    perf = performance(labelled, float(stats["test_recall"]), cfg)
    alerts = build_alerts(
        daily=daily,
        perf=perf,
        invalid_fraction=requests["invalid_fraction"],
        error_rate=requests["error_rate"],
        cfg=cfg,
    )
    return {
        "generated_for_days": int(cfg["lookback_days"]),
        "n_predictions": len(predictions),
        "service": requests,
        "daily": daily,
        "performance": perf,
        "baselines": stats,
        "alerts": alerts,
    }


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    params = load_params()
    cfg = params["monitoring"]
    pool = open_pool(os.environ["DATABASE_URL"])
    try:
        init_schema(pool)
        days = int(cfg["lookback_days"])
        predictions = read_predictions(pool, days)
        reference = pd.read_parquet(REFERENCE_PATH)
        stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        summary = build_summary(predictions, reference, stats, read_request_stats(pool, days), cfg)
        html = None
        if len(predictions) >= int(cfg["min_rows_per_day"]):
            columns = [*cfg["numeric_features"], *cfg["categorical_features"]]
            cap = int(cfg["report_sample_rows"])
            seed = int(params["seed"])
            html = evidently_html(
                reference.sample(min(cap, len(reference)), random_state=seed),
                predictions.sample(min(cap, len(predictions)), random_state=seed),
                columns,
            )
        save_monitoring_report(pool, "nightly", summary, html, keep=int(cfg["keep_reports"]))
    finally:
        close_pool(pool)
    print(json.dumps({"n_predictions": summary["n_predictions"], "alerts": summary["alerts"]}))
    return 1 if summary["alerts"] else 0


if __name__ == "__main__":
    sys.exit(main())
