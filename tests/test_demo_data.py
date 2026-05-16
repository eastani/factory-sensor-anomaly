"""Tests for the demo-data button helper."""

from __future__ import annotations

import json
from typing import Any

import httpx

from factory_anomaly.dashboard.client import ApiClient
from factory_anomaly.dashboard.demo_data import DEMO_SENSOR_NAME, generate_and_send


def test_generate_and_send_calls_ingest_then_infer() -> None:
    requests: list[tuple[str, str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content.decode()) if request.content else {}
        requests.append((request.method, request.url.path, body))
        if request.url.path == "/ingest":
            return httpx.Response(201, json={"inserted": len(body["readings"])})
        if request.url.path == "/infer":
            return httpx.Response(
                201,
                json={
                    "timestamp": "2026-05-16T00:00:00Z",
                    "machine_id": body["machine_id"],
                    "window_start": "2026-05-16T00:00:00Z",
                    "window_end": "2026-05-16T00:00:00Z",
                    "score": 0.42,
                    "is_anomaly": True,
                    "model_name": "isolation_forest",
                    "model_version": "fake-v1",
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = ApiClient(base_url="http://api.test")
    client._client = httpx.Client(transport=transport, base_url="http://api.test")

    summary = generate_and_send(client, "pump-1", n_samples=40, with_spikes=True, seed=7)

    assert [r[1] for r in requests] == ["/ingest", "/infer"]
    ingest_body = requests[0][2]
    assert ingest_body["machine_id"] == "pump-1"
    assert len(ingest_body["readings"]) == 40
    assert all(r["sensor_name"] == DEMO_SENSOR_NAME for r in ingest_body["readings"])

    assert summary["ingested"] == 40
    assert summary["score"] == 0.42
    assert summary["is_anomaly"] is True
    assert summary["model_version"] == "fake-v1"


def test_generate_and_send_without_spikes_omits_spike_step() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/ingest":
            body = json.loads(request.content.decode())
            return httpx.Response(201, json={"inserted": len(body["readings"])})
        return httpx.Response(
            201,
            json={
                "timestamp": "2026-05-16T00:00:00Z",
                "machine_id": "x",
                "window_start": "2026-05-16T00:00:00Z",
                "window_end": "2026-05-16T00:00:00Z",
                "score": 0.0,
                "is_anomaly": False,
                "model_name": "isolation_forest",
                "model_version": "v0",
            },
        )

    transport = httpx.MockTransport(handler)
    client = ApiClient(base_url="http://api.test")
    client._client = httpx.Client(transport=transport, base_url="http://api.test")

    summary = generate_and_send(client, "pump-1", n_samples=30, with_spikes=False, seed=1)
    assert summary["ingested"] == 30
