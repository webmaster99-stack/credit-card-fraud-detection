"""Monitoring's Postgres paths: request log, stored reports and the nightly job's queries.

Skipped without TEST_DATABASE_URL, like the other API DB tests (see `pg_pool` in conftest.py).
"""

from api.db import insert_prediction, latest_monitoring_report, log_request, save_monitoring_report
from monitoring.nightly import read_predictions, read_request_stats
from tests.test_api_db import ROW, SCORES


def _reset(pool) -> None:
    with pool.connection() as conn:
        conn.execute("TRUNCATE api_requests, monitoring_reports")


def test_request_stats_count_errors_and_invalid_inputs(pg_pool, clean_predictions) -> None:
    _reset(pg_pool)
    for status in (200, 200, 422, 500):
        log_request(pg_pool, "/v1/predict", status, 12.0)
    log_request(pg_pool, "/v1/model", 500, 1.0)  # not a scoring path: ignored
    stats = read_request_stats(pg_pool, days=1)
    assert stats["requests"] == 4
    assert stats["error_rate"] == 0.25
    assert stats["invalid_fraction"] == 0.25
    assert stats["p95_latency_ms"] == 12.0


def test_request_stats_empty(pg_pool, clean_predictions) -> None:
    _reset(pg_pool)
    stats = read_request_stats(pg_pool, days=1)
    assert stats["requests"] == 0 and stats["error_rate"] is None


def test_save_and_latest_report_trims_to_keep(pg_pool) -> None:
    _reset(pg_pool)
    for i in range(3):
        save_monitoring_report(pg_pool, "nightly", {"i": i}, f"<p>{i}</p>", keep=2)
    latest = latest_monitoring_report(pg_pool, "nightly", with_html=True)
    assert latest is not None
    assert latest["summary"] == {"i": 2} and latest["report_html"] == "<p>2</p>"
    with pg_pool.connection() as conn:
        (count,) = conn.execute("SELECT count(*) FROM monitoring_reports").fetchone()
    assert count == 2
    assert latest_monitoring_report(pg_pool, "replay") is None


def test_read_predictions_has_day_and_labels(pg_pool, clean_predictions) -> None:
    request_id = insert_prediction(pg_pool, ROW, SCORES, source="single")
    with pg_pool.connection() as conn:
        conn.execute(
            "UPDATE predictions SET true_label = true WHERE request_id = %s", (request_id,)
        )
    frame = read_predictions(pg_pool, days=1)
    assert len(frame) == 1
    assert frame["true_label"].iloc[0]
    assert len(frame["day"].iloc[0]) == 10
