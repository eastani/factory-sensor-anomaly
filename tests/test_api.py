"""Integration tests for the FastAPI service.

These exercise the full stack — real Postgres via testcontainers, a real
trained model loaded by the lifespan handler. No mocks.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def _make_payload(
    machine_id: str = "pump-001",
    sensor_name: str = "sensor_00",
    count: int = 30,
    start_value: float = 0.0,
) -> dict[str, Any]:
    base = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
    return {
        "machine_id": machine_id,
        "readings": [
            {
                "timestamp": (base + timedelta(seconds=i)).isoformat(),
                "sensor_name": sensor_name,
                "value": start_value + 0.1 * i,
            }
            for i in range(count)
        ],
    }


def test_healthz_reports_db_ok_and_model_loaded(api_client: TestClient) -> None:
    response = api_client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"
    assert body["model"] == "loaded"
    assert body["model_version"] == "test-v1"


def test_openapi_schema_is_served(api_client: TestClient) -> None:
    response = api_client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/ingest" in paths
    assert "/infer" in paths
    assert "/healthz" in paths


def test_ingest_persists_readings_and_returns_count(api_client: TestClient) -> None:
    payload = _make_payload(count=5)
    response = api_client.post("/ingest", json=payload)
    assert response.status_code == 201, response.text
    assert response.json() == {"inserted": 5}


def test_ingest_rejects_empty_readings_list(api_client: TestClient) -> None:
    response = api_client.post("/ingest", json={"machine_id": "x", "readings": []})
    assert response.status_code == 422


def test_readings_round_trip_via_api(api_client: TestClient) -> None:
    api_client.post("/ingest", json=_make_payload(machine_id="pump-rt", count=10))
    response = api_client.get("/readings/pump-rt", params={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    # Newest-first ordering by API contract.
    timestamps = [r["timestamp"] for r in body]
    assert timestamps == sorted(timestamps, reverse=True)


def test_readings_filter_by_sensor_name(api_client: TestClient) -> None:
    api_client.post("/ingest", json=_make_payload(machine_id="pump-f", sensor_name="a", count=3))
    api_client.post("/ingest", json=_make_payload(machine_id="pump-f", sensor_name="b", count=2))
    response = api_client.get("/readings/pump-f", params={"sensor_name": "b"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert all(r["sensor_name"] == "b" for r in body)


def test_infer_returns_score_and_persists_anomaly_result(api_client: TestClient) -> None:
    api_client.post("/ingest", json=_make_payload(machine_id="pump-inf", count=30))
    response = api_client.post(
        "/infer",
        json={
            "machine_id": "pump-inf",
            "sensor_name": "sensor_00",
            "window_size": 20,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["machine_id"] == "pump-inf"
    assert isinstance(body["score"], float)
    assert isinstance(body["is_anomaly"], bool)
    assert body["model_name"] == "isolation_forest"
    assert body["model_version"] == "test-v1"

    # The result is persisted and retrievable.
    listed = api_client.get("/anomalies/pump-inf").json()
    assert len(listed) >= 1


def test_infer_returns_422_when_not_enough_readings(api_client: TestClient) -> None:
    api_client.post("/ingest", json=_make_payload(machine_id="pump-short", count=5))
    response = api_client.post(
        "/infer",
        json={"machine_id": "pump-short", "sensor_name": "sensor_00", "window_size": 20},
    )
    assert response.status_code == 422
    assert "found 5" in response.json()["detail"]


def test_anomalies_only_filter(api_client: TestClient) -> None:
    # Just verify the param is accepted and the response shape is correct.
    response = api_client.get("/anomalies/pump-001", params={"only_anomalies": "true", "limit": 10})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
