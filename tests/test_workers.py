"""Unit tests for the ingester and scorer workers.

The ``main()`` loops themselves are excluded from coverage (pragma: no cover)
and exercised end-to-end via ``docker compose up`` and CI. What we *do* unit
test here is the ``run_one_tick`` function — the per-iteration unit of work
— so behavioural changes (e.g. "scorer must not crash on 503") are caught
without spinning up the stack.
"""

from __future__ import annotations

import json

import httpx
import pytest

from factory_anomaly.dashboard.client import ApiClient
from factory_anomaly.workers.ingester import IngesterConfig
from factory_anomaly.workers.ingester import run_one_tick as ingester_tick
from factory_anomaly.workers.scorer import ScorerConfig
from factory_anomaly.workers.scorer import run_one_tick as scorer_tick


def _client(handler: httpx.MockTransport) -> ApiClient:
    client = ApiClient(base_url="http://api.test")
    client._client = httpx.Client(transport=handler, base_url="http://api.test")
    return client


# ------------------------------ ingester ---------------------------------


def test_ingester_posts_expected_payload_shape() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ingest"
        captured.update(json.loads(request.content.decode()))
        return httpx.Response(201, json={"inserted": 12})

    client = _client(httpx.MockTransport(handler))
    cfg = IngesterConfig(
        machine_id="pump-x",
        sensor_name="vibration",
        batch_size=12,
        spike_probability=0.0,
    )
    assert ingester_tick(client, cfg, tick=1, seed=99) is True
    assert captured["machine_id"] == "pump-x"
    readings = captured["readings"]
    assert isinstance(readings, list)
    assert len(readings) == 12
    assert all(r["sensor_name"] == "vibration" for r in readings)


def test_ingester_returns_false_on_api_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _client(httpx.MockTransport(handler))
    cfg = IngesterConfig(batch_size=5, spike_probability=0.0)
    assert ingester_tick(client, cfg, tick=1, seed=1) is False


def test_ingester_config_reads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGESTER_MACHINE_ID", "pump-env")
    monkeypatch.setenv("INGESTER_BATCH_SIZE", "7")
    monkeypatch.setenv("INGESTER_INTERVAL_SECONDS", "3.5")
    cfg = IngesterConfig.from_env()
    assert cfg.machine_id == "pump-env"
    assert cfg.batch_size == 7
    assert cfg.interval_seconds == pytest.approx(3.5)


# ------------------------------ scorer -----------------------------------


def test_scorer_records_successful_inference() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode()))
        return httpx.Response(
            201,
            json={
                "timestamp": "2026-05-16T00:00:00Z",
                "machine_id": "pump-001",
                "window_start": "2026-05-16T00:00:00Z",
                "window_end": "2026-05-16T00:00:00Z",
                "score": 0.7,
                "is_anomaly": True,
                "model_name": "isolation_forest",
                "model_version": "test-v1",
            },
        )

    client = _client(httpx.MockTransport(handler))
    cfg = ScorerConfig(machine_id="pump-001", sensor_name="sensor_00", window_size=20)
    assert scorer_tick(client, cfg, tick=1) is True
    assert captured["window_size"] == 20


def test_scorer_treats_422_as_non_fatal_transient() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text="not enough readings")

    client = _client(httpx.MockTransport(handler))
    cfg = ScorerConfig()
    assert scorer_tick(client, cfg, tick=1) is False  # boolean signals "did not score"


def test_scorer_treats_503_as_non_fatal_transient() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model unavailable")

    client = _client(httpx.MockTransport(handler))
    cfg = ScorerConfig()
    assert scorer_tick(client, cfg, tick=1) is False


def test_scorer_treats_5xx_other_than_503_as_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _client(httpx.MockTransport(handler))
    cfg = ScorerConfig()
    # Still does not raise — workers must keep running.
    assert scorer_tick(client, cfg, tick=1) is False


def test_scorer_config_reads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCORER_MACHINE_ID", "pump-z")
    monkeypatch.setenv("SCORER_INTERVAL_SECONDS", "2.5")
    monkeypatch.setenv("SCORER_WINDOW_SIZE", "40")
    cfg = ScorerConfig.from_env()
    assert cfg.machine_id == "pump-z"
    assert cfg.window_size == 40
    assert cfg.interval_seconds == pytest.approx(2.5)
