"""Unit tests for the dashboard's API client.

These do not require a running FastAPI server — ``httpx.MockTransport`` lets
us assert the wire format the client sends and stub the response in-process.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from factory_anomaly.dashboard.client import ApiClient, ApiClientError


def _client_with_handler(handler: Callable[[httpx.Request], httpx.Response]) -> ApiClient:
    """Build an ApiClient wired to an in-process MockTransport."""
    transport = httpx.MockTransport(handler)
    client = ApiClient(base_url="http://api.test")
    client._client = httpx.Client(transport=transport, base_url="http://api.test")
    return client


def test_healthz_round_trip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/healthz"
        return httpx.Response(200, json={"status": "ok", "db": "ok", "model": "loaded"})

    with _client_with_handler(handler) as client:
        body = client.healthz()
    assert body["status"] == "ok"


def test_list_readings_passes_query_params() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/readings/pump-1"
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=[])

    with _client_with_handler(handler) as client:
        client.list_readings("pump-1", sensor_name="a", limit=42)

    assert captured == {"sensor_name": "a", "limit": "42"}


def test_list_readings_omits_sensor_when_none() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=[])

    with _client_with_handler(handler) as client:
        client.list_readings("pump-1", limit=5)

    assert "sensor_name" not in captured
    assert captured["limit"] == "5"


def test_list_anomalies_serialises_only_anomalies_bool_as_lowercase() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=[])

    with _client_with_handler(handler) as client:
        client.list_anomalies("pump-1", only_anomalies=True)
    assert captured["only_anomalies"] == "true"


def test_ingest_sends_expected_body() -> None:
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content.decode()))
        return httpx.Response(201, json={"inserted": 3})

    readings = [{"timestamp": "2026-05-16T00:00:00Z", "sensor_name": "a", "value": 1.0}]
    with _client_with_handler(handler) as client:
        body = client.ingest("pump-1", readings)

    assert body == {"inserted": 3}
    assert sent == {"machine_id": "pump-1", "readings": readings}


def test_infer_passes_window_size_only_when_set() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(201, json={"score": 0.1})

    with _client_with_handler(handler) as client:
        client.infer("pump-1", "a")
        client.infer("pump-1", "a", window_size=30)

    assert "window_size" not in bodies[0]
    assert bodies[1]["window_size"] == 30


def test_4xx_response_raises_with_status_code() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text="too few samples")

    with _client_with_handler(handler) as client, pytest.raises(ApiClientError) as info:
        client.infer("pump-1", "a")
    assert info.value.status_code == 422
    assert "422" in str(info.value)


def test_network_error_is_normalised() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with _client_with_handler(handler) as client, pytest.raises(ApiClientError) as info:
        client.healthz()
    assert info.value.status_code is None
    assert "failed" in str(info.value)
