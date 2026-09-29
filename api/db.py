"""Postgres access: the prediction log doubles as the online per-card history store.

One connection pool per process (`init_pool`/`close_pool`, called from the app's lifespan). Every
function takes the pool explicitly rather than reaching for a global, so tests can pass a pool
pointed at a throwaway database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from fraud.serving.schema import REQUIRED_COLUMNS, validate_transactions

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# Columns stored per prediction, beyond the transaction fields themselves.
_SCORE_COLUMNS = ["fraud_probability", "flagged", "model_name", "model_version", "pipeline_version"]


def open_pool(database_url: str, *, min_size: int = 1, max_size: int = 5) -> ConnectionPool:
    # Neon closes idle connections (scale-to-zero, pooler timeouts), so test each connection as it
    # is handed out and replace a dead one; max_idle retires connections before Neon does.
    pool = ConnectionPool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        max_idle=60,
        check=ConnectionPool.check_connection,
        open=False,
    )
    pool.open(wait=True, timeout=10)
    return pool


def close_pool(pool: ConnectionPool) -> None:
    pool.close()


def init_schema(pool: ConnectionPool) -> None:
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    with pool.connection() as conn:
        conn.execute(ddl)


def ping(pool: ConnectionPool) -> bool:
    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _native(value: Any) -> Any:
    """A DB-bindable Python value for a pandas/numpy scalar."""
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def insert_prediction(
    pool: ConnectionPool,
    row: dict[str, Any],
    scores: dict[str, Any],
    *,
    source: str = "single",
) -> UUID:
    """Store one scored transaction; returns its `request_id`."""
    columns = [*REQUIRED_COLUMNS, "card_id", *_SCORE_COLUMNS, "source"]
    values = {c: _native(row[c]) for c in REQUIRED_COLUMNS}
    values["card_id"] = row["card_id"]
    values.update({c: _native(scores[c]) for c in _SCORE_COLUMNS})
    values["source"] = source
    placeholders = ", ".join(f"%({c})s" for c in columns)
    query = (
        f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({placeholders}) "  # noqa: S608
        "RETURNING request_id"
    )
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(query, values)
        result = cur.fetchone()
        assert result is not None
        request_id: UUID = result[0]
        return request_id


def insert_predictions_batch(
    pool: ConnectionPool,
    rows: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    source: str = "batch",
) -> list[UUID]:
    """Store many scored transactions in one round trip; returns their `request_id`s, row order."""
    columns = [*REQUIRED_COLUMNS, "card_id", *_SCORE_COLUMNS, "source"]
    placeholders = ", ".join(f"%({c})s" for c in columns)
    query = (
        f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({placeholders}) "  # noqa: S608
        "RETURNING request_id"
    )
    params_seq = []
    for row, score in zip(rows, scores, strict=True):
        values = {c: _native(row[c]) for c in REQUIRED_COLUMNS}
        values["card_id"] = row["card_id"]
        values.update({c: _native(score[c]) for c in _SCORE_COLUMNS})
        values["source"] = source
        params_seq.append(values)

    request_ids: list[UUID] = []
    with pool.connection() as conn, conn.cursor() as cur:
        cur.executemany(query, params_seq, returning=True)
        while True:
            result = cur.fetchone()
            assert result is not None
            request_ids.append(result[0])
            if not cur.nextset():
                break
    return request_ids


def fetch_card_history(
    pool: ConnectionPool,
    card_id: str,
    before: datetime,
    *,
    max_rows: int = 2000,
) -> pd.DataFrame:
    """Every earlier stored transaction for ``card_id`` (strictly before ``before``), time-ordered.

    Columns match the v1 input schema plus `card_id`, so the result can be passed straight to
    `fraud.features.pipeline.transform_with_context` as history. Re-validated on the way out so its
    dtypes (`trans_ts`, `dob`, ...) match a freshly-submitted request row exactly: psycopg hands
    back plain `date`/`datetime` objects that, left uncoerced, would give pandas a mixed-dtype
    column once concatenated with the new row and break datetime arithmetic.
    """
    columns = [*REQUIRED_COLUMNS, "card_id"]
    query = (
        f"SELECT {', '.join(columns)} FROM predictions "  # noqa: S608
        "WHERE card_id = %(card_id)s AND trans_ts < %(before)s "
        "ORDER BY trans_ts DESC LIMIT %(max_rows)s"
    )
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            query, {"card_id": card_id, "before": before, "max_rows": max_rows}
        ).fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    validated = validate_transactions(frame)
    return validated.sort_values("trans_ts").reset_index(drop=True)


def record_feedback(pool: ConnectionPool, request_id: UUID, is_fraud: bool) -> bool:
    """Attach a true label to a past prediction; False if `request_id` does not exist."""
    query = (
        "UPDATE predictions SET true_label = %(is_fraud)s, label_recorded_at = %(now)s "
        "WHERE request_id = %(request_id)s"
    )
    params = {"is_fraud": is_fraud, "now": datetime.now(UTC), "request_id": request_id}
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.rowcount > 0


def log_request(pool: ConnectionPool, path: str, status_code: int, duration_ms: float) -> None:
    """Best-effort record of one API request for the service/data-quality monitoring layers."""
    try:
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO api_requests (path, status_code, duration_ms) VALUES (%s, %s, %s)",
                (path, status_code, duration_ms),
            )
    except Exception:  # monitoring must never break serving
        return


def save_monitoring_report(
    pool: ConnectionPool,
    kind: str,
    summary: dict[str, Any],
    report_html: str | None = None,
    *,
    keep: int = 60,
) -> None:
    """Store one job result, then trim that kind to its ``keep`` newest rows."""
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO monitoring_reports (kind, summary, report_html) VALUES (%s, %s, %s)",
            (kind, Jsonb(summary), report_html),
        )
        conn.execute(
            "DELETE FROM monitoring_reports WHERE kind = %(kind)s AND id NOT IN ("
            "SELECT id FROM monitoring_reports WHERE kind = %(kind)s "
            "ORDER BY created_at DESC LIMIT %(keep)s)",
            {"kind": kind, "keep": keep},
        )


def latest_monitoring_report(
    pool: ConnectionPool, kind: str, *, with_html: bool = False
) -> dict[str, Any] | None:
    columns = "created_at, summary" + (", report_html" if with_html else "")
    query = (
        f"SELECT {columns} FROM monitoring_reports "  # noqa: S608
        "WHERE kind = %s ORDER BY created_at DESC LIMIT 1"
    )
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        row: dict[str, Any] | None = cur.execute(query, (kind,)).fetchone()
    return row


__all__ = [
    "close_pool",
    "fetch_card_history",
    "latest_monitoring_report",
    "log_request",
    "save_monitoring_report",
    "init_schema",
    "insert_prediction",
    "insert_predictions_batch",
    "open_pool",
    "ping",
    "record_feedback",
]
