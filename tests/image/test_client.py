"""Tests for the image-API HTTP client.

Uses ``httpx.MockTransport`` instead of spinning up a real server — the
client is a pure HTTP wrapper, no torch required, fast to test.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from factory_anomaly.image.client import ImageApiClient, ImageApiClientError

Handler = Callable[[httpx.Request], httpx.Response]


def _client_with_handler(handler: Handler) -> ImageApiClient:
    transport = httpx.MockTransport(handler)
    api = ImageApiClient("http://test")
    # Swap the underlying httpx.Client for one using the mock transport.
    api._client.close()
    api._client = httpx.Client(base_url="http://test", transport=transport, timeout=1.0)
    return api


def test_healthz_parses_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/healthz"
        return httpx.Response(200, json={"status": "ok", "model_loaded": True})

    with _client_with_handler(handler) as client:
        assert client.healthz()["status"] == "ok"


def test_predict_sends_multipart_and_returns_json() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["content_type"] = request.headers["content-type"]
        captured["body_len"] = len(request.content)
        return httpx.Response(200, json={"score": 0.42, "model_version": "v"})

    with _client_with_handler(handler) as client:
        result = client.predict(b"fake-png-bytes", filename="example.png")
        assert result["score"] == 0.42

    assert captured["path"] == "/image/predict"
    assert "multipart/form-data" in str(captured["content_type"])
    body_len = captured["body_len"]
    assert isinstance(body_len, int)
    assert body_len > 0


def test_non_2xx_raises_with_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model not loaded")

    with _client_with_handler(handler) as client, pytest.raises(ImageApiClientError) as exc:
        client.predict(b"x")
    assert exc.value.status_code == 503
    assert "503" in str(exc.value)


def test_transport_error_wrapped() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns")

    with _client_with_handler(handler) as client, pytest.raises(ImageApiClientError):
        client.healthz()


def test_base_url_is_normalised() -> None:
    client = ImageApiClient("http://example.com/")
    assert client.base_url == "http://example.com"
    client.close()
