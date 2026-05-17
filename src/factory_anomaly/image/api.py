"""FastAPI app for the image-anomaly modality (Phase 3.2; see ADR-0006).

This module deliberately lives next to the rest of the image code rather
than under ``factory_anomaly.api`` so that the default api package stays
torch-free and can be imported by the sensor service without dragging
PyTorch into its container.

The PatchCore memory bank is loaded once at startup via the lifespan
handler. ``/healthz`` reports the readiness of the loaded model; the
service refuses to score requests until the bank is in place.
"""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.responses import Response

from factory_anomaly.config import get_image_api_settings
from factory_anomaly.image.detector import (
    PatchCoreDetector,
    PatchCoreVersionMismatchError,
)
from factory_anomaly.logging_config import configure_logging, get_logger
from factory_anomaly.observability import (
    IMAGE_INFERENCE_DURATION,
    IMAGE_INFERENCE_SCORE,
    IMAGE_INFERENCES_TOTAL,
    IMAGE_MODEL_LOADED,
    metrics_response,
    register_metrics_middleware,
)

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ImageHealthResponse(BaseModel):
    status: str = Field(description="``ok`` if scoring is possible, else ``degraded``.")
    model_loaded: bool
    model_version: str | None = None


class ImagePredictResponse(BaseModel):
    model_version: str
    score: float = Field(
        description="Max per-patch nearest-neighbour distance; higher = more anomalous."
    )
    anomaly_map: list[list[float]] = Field(
        description="2-D per-patch anomaly scores (H x W = target_spatial x target_spatial)."
    )
    width: int
    height: int
    elapsed_ms: float = Field(
        description="Server-side scoring time, excluding image decode and network."
    )


# ---------------------------------------------------------------------------
# Lifespan + factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the PatchCore memory bank once at startup.

    The model artifact is **required** for this service to be useful — unlike
    the sensor API, which can run degraded with a missing model, the image
    service has nothing meaningful to return. We still keep the process up so
    operators can see structured logs and Prometheus metrics, but ``/healthz``
    reports ``degraded`` and ``/image/predict`` returns 503.
    """
    settings = get_image_api_settings()
    configure_logging(settings.log_level)

    # Cap CPU threads (see ADR-0006 implementation notes). Doing this before
    # the detector loads ensures all downstream torch ops respect the cap.
    torch.set_num_threads(settings.torch_num_threads)
    log.info("startup.torch_threads_capped", n=settings.torch_num_threads)

    try:
        detector = PatchCoreDetector.load(settings.model_path)
        app.state.detector = detector
        IMAGE_MODEL_LOADED.labels(model_version=detector.metadata.model_version).set(1)
        log.info(
            "startup.image_model_loaded",
            path=str(settings.model_path),
            version=detector.metadata.model_version,
            torch_version=detector.metadata.torch_version,
            memory_bank_shape=detector.metadata.memory_bank_shape,
        )
    except (FileNotFoundError, PatchCoreVersionMismatchError) as exc:
        app.state.detector = None
        IMAGE_MODEL_LOADED.labels(model_version="none").set(0)
        log.warning(
            "startup.image_model_unavailable",
            path=str(settings.model_path),
            reason=exc.__class__.__name__,
            message=str(exc),
        )

    try:
        yield
    finally:
        log.info("shutdown.image_api")


def create_app() -> FastAPI:
    """Build a fresh FastAPI app for the image-anomaly service."""
    app = FastAPI(
        title="factory-sensor-anomaly · image",
        description=(
            "Unsupervised image anomaly detection (PatchCore over a frozen "
            "ResNet50 backbone). See ADR-0006."
        ),
        version="0.0.1",
        lifespan=lifespan,
    )
    register_metrics_middleware(app)

    @app.get("/healthz", response_model=ImageHealthResponse, tags=["meta"])
    def healthz(request: Request) -> ImageHealthResponse:
        detector: PatchCoreDetector | None = getattr(request.app.state, "detector", None)
        if detector is None:
            return ImageHealthResponse(status="degraded", model_loaded=False)
        return ImageHealthResponse(
            status="ok",
            model_loaded=True,
            model_version=detector.metadata.model_version,
        )

    @app.post("/image/predict", response_model=ImagePredictResponse, tags=["inference"])
    async def predict(request: Request, image: UploadFile) -> ImagePredictResponse:
        detector: PatchCoreDetector | None = getattr(request.app.state, "detector", None)
        if detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="memory bank not loaded; check service startup logs",
            )

        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")

        settings = get_image_api_settings()
        try:
            tensor = _decode_to_tensor(raw, settings.input_size)
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc

        version = detector.metadata.model_version
        start = time.perf_counter()
        score, anomaly_map = detector.score(tensor)
        elapsed = time.perf_counter() - start

        IMAGE_INFERENCES_TOTAL.labels(model_version=version).inc()
        IMAGE_INFERENCE_DURATION.labels(model_version=version).observe(elapsed)
        IMAGE_INFERENCE_SCORE.labels(model_version=version).observe(score)

        log.info(
            "image.predict",
            model_version=version,
            score=score,
            elapsed_ms=elapsed * 1000,
            map_shape=anomaly_map.shape,
        )

        return ImagePredictResponse(
            model_version=version,
            score=score,
            anomaly_map=anomaly_map.tolist(),
            width=int(anomaly_map.shape[1]),
            height=int(anomaly_map.shape[0]),
            elapsed_ms=elapsed * 1000,
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return metrics_response()

    return app


def _decode_to_tensor(raw: bytes, input_size: int) -> torch.Tensor:
    """Decode an uploaded image into a ``(1, 3, S, S)`` normalised tensor.

    Resize to a fixed ``input_size`` (default 224) so the ResNet50 backbone
    always sees its training-time resolution — variable input sizes work
    too, but pinning makes latency and feature-map shapes predictable.
    """
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return PatchCoreDetector.stack_images([arr])


# Module-level ASGI handle for ``uvicorn factory_anomaly.image.api:app``.
app = create_app()
