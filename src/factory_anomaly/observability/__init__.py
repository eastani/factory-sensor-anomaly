"""Observability primitives — Prometheus metrics + middleware."""

from factory_anomaly.observability.metrics import (
    IMAGE_INFERENCE_DURATION,
    IMAGE_INFERENCE_SCORE,
    IMAGE_INFERENCES_TOTAL,
    IMAGE_MODEL_LOADED,
    INFERENCE_DURATION,
    INFERENCE_SCORE,
    INFERENCES_TOTAL,
    INGEST_ROWS_TOTAL,
    MODEL_LOADED,
    metrics_response,
)
from factory_anomaly.observability.middleware import register_metrics_middleware

__all__ = [
    "IMAGE_INFERENCES_TOTAL",
    "IMAGE_INFERENCE_DURATION",
    "IMAGE_INFERENCE_SCORE",
    "IMAGE_MODEL_LOADED",
    "INFERENCES_TOTAL",
    "INFERENCE_DURATION",
    "INFERENCE_SCORE",
    "INGEST_ROWS_TOTAL",
    "MODEL_LOADED",
    "metrics_response",
    "register_metrics_middleware",
]
