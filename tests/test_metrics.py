"""Tests for the Prometheus /metrics endpoint and the route counters.

Counter / histogram singletons are module-globals — running the full integration
test in any order will mutate them. We don't reset between tests; we assert
*relative* growth instead.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def _read_counter(metrics_text: str, name: str, **labels: str) -> float:
    """Parse a Prometheus exposition payload and return one labelled counter."""
    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        pattern = rf"^{re.escape(name)}\{{{re.escape(label_str)}\}} (\S+)"
    else:
        pattern = rf"^{re.escape(name)} (\S+)"
    for line in metrics_text.splitlines():
        match = re.match(pattern, line)
        if match:
            return float(match.group(1))
    return 0.0


def test_metrics_endpoint_serves_prometheus_format(api_client: TestClient) -> None:
    response = api_client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    # Sanity: at least a few of our named metrics appear.
    assert "factory_anomaly_http_requests_total" in body
    assert "factory_anomaly_model_loaded" in body


def test_metrics_endpoint_is_not_in_openapi(api_client: TestClient) -> None:
    schema = api_client.get("/openapi.json").json()
    assert "/metrics" not in schema["paths"]


def test_ingest_increments_rows_counter(api_client: TestClient) -> None:
    before = _read_counter(api_client.get("/metrics").text, "factory_anomaly_ingest_rows_total")
    payload = {
        "machine_id": "metrics-test",
        "readings": [
            {"timestamp": "2026-05-16T00:00:00Z", "sensor_name": "s", "value": 1.0},
            {"timestamp": "2026-05-16T00:00:01Z", "sensor_name": "s", "value": 1.1},
            {"timestamp": "2026-05-16T00:00:02Z", "sensor_name": "s", "value": 1.2},
        ],
    }
    response = api_client.post("/ingest", json=payload)
    assert response.status_code == 201

    after = _read_counter(api_client.get("/metrics").text, "factory_anomaly_ingest_rows_total")
    assert after - before == pytest.approx(3.0)


def test_http_request_counter_increments_with_route_template(api_client: TestClient) -> None:
    # Hit /healthz a couple of times to drive the counter up.
    for _ in range(3):
        api_client.get("/healthz")

    body = api_client.get("/metrics").text
    count = _read_counter(
        body,
        "factory_anomaly_http_requests_total",
        method="GET",
        route="/healthz",
        status_class="2xx",
    )
    assert count >= 3


def test_model_loaded_gauge_reflects_test_model(api_client: TestClient) -> None:
    body = api_client.get("/metrics").text
    value = _read_counter(
        body,
        "factory_anomaly_model_loaded",
        model_version="test-v1",
    )
    assert value == 1.0
