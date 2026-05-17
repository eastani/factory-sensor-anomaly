"""Tests for the image-anomaly FastAPI service."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _png_bytes(size: int = 64, *, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_healthz_reports_loaded_model(image_api_client: TestClient) -> None:
    response = image_api_client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-image-v1"


def test_predict_returns_score_and_heatmap(image_api_client: TestClient) -> None:
    response = image_api_client.post(
        "/image/predict",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_version"] == "test-image-v1"
    assert isinstance(body["score"], float)
    assert body["score"] >= 0.0
    assert body["width"] == 4
    assert body["height"] == 4
    assert len(body["anomaly_map"]) == 4
    assert all(len(row) == 4 for row in body["anomaly_map"])
    assert body["elapsed_ms"] >= 0.0


def test_predict_max_of_map_equals_score(image_api_client: TestClient) -> None:
    response = image_api_client.post(
        "/image/predict",
        files={"image": ("test.png", _png_bytes(seed=7), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    flat = [v for row in body["anomaly_map"] for v in row]
    assert max(flat) == pytest.approx(body["score"])


def test_predict_rejects_empty_upload(image_api_client: TestClient) -> None:
    response = image_api_client.post(
        "/image/predict",
        files={"image": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_rejects_garbage(image_api_client: TestClient) -> None:
    response = image_api_client.post(
        "/image/predict",
        files={"image": ("garbage.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_predict_accepts_jpeg(image_api_client: TestClient) -> None:
    """The decoder uses PIL.Image.open, which auto-detects format."""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    response = image_api_client.post(
        "/image/predict",
        files={"image": ("test.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200


def test_metrics_endpoint_exposes_image_counters(image_api_client: TestClient) -> None:
    # Drive at least one inference so the counter is non-zero.
    image_api_client.post(
        "/image/predict",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )

    response = image_api_client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "factory_anomaly_image_inferences_total" in body
    assert "factory_anomaly_image_inference_duration_seconds" in body
    assert "factory_anomaly_image_model_loaded" in body


def test_healthz_degraded_when_model_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempdirFactory
) -> None:
    """If IMAGE_API_MODEL_PATH points to a missing file, startup logs a
    warning and /healthz reports degraded with 200 (process stays up)."""
    missing = tmp_path / "nope.joblib"  # type: ignore[operator]
    monkeypatch.setenv("IMAGE_API_MODEL_PATH", str(missing))

    from factory_anomaly.config import get_image_api_settings

    get_image_api_settings.cache_clear()

    from factory_anomaly.image.api import create_app

    app = create_app()
    try:
        with TestClient(app) as client:
            response = client.get("/healthz")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "degraded"
            assert body["model_loaded"] is False

            predict = client.post(
                "/image/predict",
                files={"image": ("test.png", _png_bytes(), "image/png")},
            )
            assert predict.status_code == 503
    finally:
        get_image_api_settings.cache_clear()
