"""FastAPI endpoint tests.

`TestClient(app)` used *without* `with` never runs the app's lifespan (verified: it only fires
inside a `with` block), so these tests never make a real HF Hub download or a real database
connection just by importing/calling the app - `get_model`/`get_pool` are overridden instead.
Tests that need a real database use the `pg_pool` fixture and skip cleanly without one.
"""

import io

import pytest
from api.deps import get_model, get_pool
from api.main import app
from fastapi.testclient import TestClient

from fraud.serving import load_model

WRONG_HEADERS = {"X-API-Key": "wrong"}
HEADERS = {"X-API-Key": "test-api-key"}  # matches tests/conftest.py's API_KEY default

TXN = {
    "trans_ts": "2020-11-05T18:42:10",
    "amt": 54.20,
    "category": "grocery_pos",
    "gender": "F",
    "state": "IL",
    "city_pop": 116_250,
    "dob": "1985-03-02",
    "lat": 39.80,
    "long": -89.64,
    "merch_lat": 39.85,
    "merch_long": -89.70,
    "card_id": "card-api-1",
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def served_model(bundle_dir_v2):
    return load_model(str(bundle_dir_v2))


@pytest.fixture
def api_client(served_model, pg_pool, clean_predictions) -> TestClient:
    app.dependency_overrides[get_model] = lambda: served_model
    app.dependency_overrides[get_pool] = lambda: pg_pool
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_health_without_lifespan_reports_degraded(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded" and body["model_loaded"] is False


def test_predict_requires_api_key(client: TestClient) -> None:
    assert client.post("/v1/predict", json=TXN).status_code == 401


def test_predict_rejects_wrong_api_key(client: TestClient) -> None:
    resp = client.post("/v1/predict", json=TXN, headers=WRONG_HEADERS)
    assert resp.status_code == 401


def test_model_info_requires_only_the_model(served_model, client: TestClient) -> None:
    app.dependency_overrides[get_model] = lambda: served_model
    try:
        resp = client.get("/v1/model", headers=HEADERS)
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["feature_set"] == "v2"
    assert body["model_version"] == served_model.model_version


def test_predict_scores_and_logs(api_client: TestClient) -> None:
    resp = api_client.post("/v1/predict", json=TXN, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["flagged"] == (body["fraud_probability"] >= body["threshold"])
    assert body["reasons"]
    assert all(
        {"feature", "label", "value", "contribution", "direction", "text"} <= r.keys()
        for r in body["reasons"]
    )


def test_predict_uses_stored_card_history(api_client: TestClient) -> None:
    first = api_client.post("/v1/predict", json=TXN, headers=HEADERS)
    later = dict(TXN, trans_ts="2020-11-05T19:00:00", amt=999.0)
    second = api_client.post("/v1/predict", json=later, headers=HEADERS)
    assert first.status_code == second.status_code == 200
    # A second transaction minutes after the first has one prior transaction as history; a
    # stateless scoring of the same row (no context) would not see it. Just check both scored
    # without error and got distinct request ids - the history-features unit tests already pin
    # down the exact math.
    assert first.json()["request_id"] != second.json()["request_id"]


def test_predict_rejects_invalid_category(api_client: TestClient) -> None:
    bad = dict(TXN, category="not_a_real_category")
    resp = api_client.post("/v1/predict", json=bad, headers=HEADERS)
    assert resp.status_code == 422
    assert any("category" in p for p in resp.json()["detail"])


def test_predict_batch_json_list(api_client: TestClient) -> None:
    rows = [dict(TXN, card_id="card-batch-1"), dict(TXN, card_id="card-batch-2", amt=12.0)]
    resp = api_client.post("/v1/predict/batch", json=rows, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_rows"] == 2
    assert len(body["results"]) == 2
    assert len({r["request_id"] for r in body["results"]}) == 2


def test_predict_batch_missing_card_id(api_client: TestClient) -> None:
    rows = [{k: v for k, v in TXN.items() if k != "card_id"}]
    resp = api_client.post("/v1/predict/batch", json=rows, headers=HEADERS)
    assert resp.status_code == 422
    assert "card_id" in resp.json()["detail"][0]


def test_predict_batch_csv_upload(api_client: TestClient) -> None:
    header = ",".join([*[c for c in TXN if c != "card_id"], "card_id"])
    row = ",".join(str(TXN[c]) for c in TXN if c != "card_id") + f",{TXN['card_id']}"
    csv_bytes = f"{header}\n{row}\n".encode()
    resp = api_client.post(
        "/v1/predict/batch",
        headers=HEADERS,
        files={"file": ("transactions.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_rows"] == 1


def test_feedback_round_trip(api_client: TestClient) -> None:
    predicted = api_client.post("/v1/predict", json=TXN, headers=HEADERS).json()
    resp = api_client.post(
        "/v1/feedback",
        json={"request_id": predicted["request_id"], "is_fraud": True},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json() == {"request_id": predicted["request_id"], "recorded": True}


def test_feedback_unknown_request_id(api_client: TestClient) -> None:
    import uuid

    resp = api_client.post(
        "/v1/feedback",
        json={"request_id": str(uuid.uuid4()), "is_fraud": False},
        headers=HEADERS,
    )
    assert resp.status_code == 404
