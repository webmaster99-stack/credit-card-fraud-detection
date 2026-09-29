"""Unit tests for pure logic, plus integration tests against a real Postgres (skipped without
TEST_DATABASE_URL; see tests/conftest.py's `pg_pool` fixture and the repo's docker-compose.yml)."""

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from api.db import (
    _native,
    fetch_card_history,
    insert_prediction,
    insert_predictions_batch,
    ping,
    record_feedback,
)

ROW = {
    "trans_ts": pd.Timestamp("2020-11-05 18:42:10", tz="UTC"),
    "amt": 54.20,
    "category": "grocery_pos",
    "gender": "F",
    "state": "IL",
    "city_pop": 116_250,
    "dob": date(1985, 3, 2),
    "lat": 39.80,
    "long": -89.64,
    "merch_lat": 39.85,
    "merch_long": -89.70,
    "card_id": "card-1",
}
SCORES = {
    "fraud_probability": 0.12,
    "flagged": False,
    "model_name": "fraud-classifier",
    "model_version": "1",
    "pipeline_version": "features-1.0.0",
}


def test_native_converts_pandas_and_numpy_scalars() -> None:
    assert isinstance(_native(pd.Timestamp("2020-01-01")), datetime)
    assert _native(np.int64(5)) == 5 and isinstance(_native(np.int64(5)), int)
    assert _native(np.float64(1.5)) == 1.5 and isinstance(_native(np.float64(1.5)), float)
    assert _native("x") == "x"


def test_ping_false_when_unreachable() -> None:
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        "postgresql://nobody:nobody@127.0.0.1:1/nope", open=False, min_size=1, max_size=1
    )
    assert ping(pool) is False


def test_insert_and_fetch_history_round_trip(pg_pool, clean_predictions) -> None:
    insert_prediction(pg_pool, ROW, SCORES, source="single")
    later = dict(ROW, trans_ts=pd.Timestamp("2020-11-06 09:00:00", tz="UTC"), amt=10.0)
    insert_prediction(pg_pool, later, SCORES, source="single")

    history = fetch_card_history(pg_pool, "card-1", datetime(2020, 11, 7, tzinfo=UTC))
    assert list(history["amt"]) == [54.20, 10.0]  # time-ordered

    none_yet = fetch_card_history(pg_pool, "card-1", datetime(2020, 11, 5, 18, 42, 10, tzinfo=UTC))
    assert none_yet.empty  # strictly earlier: the row at exactly this timestamp is excluded


def test_insert_predictions_batch_returns_one_id_per_row(pg_pool, clean_predictions) -> None:
    rows = [ROW, dict(ROW, card_id="card-2")]
    scores = [SCORES, SCORES]
    ids = insert_predictions_batch(pg_pool, rows, scores, source="batch")
    assert len(ids) == 2 and len(set(ids)) == 2


def test_record_feedback(pg_pool, clean_predictions) -> None:
    request_id = insert_prediction(pg_pool, ROW, SCORES, source="single")
    assert record_feedback(pg_pool, request_id, True) is True
    with pg_pool.connection() as conn:
        row = conn.execute(
            "SELECT true_label FROM predictions WHERE request_id = %s", (request_id,)
        ).fetchone()
    assert row is not None and row[0] is True


def test_record_feedback_missing_request_id(pg_pool, clean_predictions) -> None:
    import uuid

    assert record_feedback(pg_pool, uuid.uuid4(), True) is False
