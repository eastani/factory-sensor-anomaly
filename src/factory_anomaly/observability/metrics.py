"""Prometheus metric singletons for the anomaly service.

Label-cardinality budget:

- ``model_version`` is bounded (one value per deployed model).
- ``status_class`` is bounded (5 values: 2xx/3xx/4xx/5xx/other).
- ``method`` is bounded (HTTP verbs).
- ``route`` uses the FastAPI route *template* (``/readings/{machine_id}``),
  not the resolved path, so per-machine cardinality does not leak in.

Anything that could explode the label set (raw machine_id, sensor name,
score buckets) lives in the value space (counter increments) or in
Postgres, not in labels.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client import (
    REGISTRY as DEFAULT_REGISTRY,
)
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Application-level metrics
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "factory_anomaly_http_requests_total",
    "Count of HTTP requests handled, labelled by route template and status class.",
    labelnames=("method", "route", "status_class"),
)

HTTP_REQUEST_DURATION = Histogram(
    "factory_anomaly_http_request_duration_seconds",
    "Wall-clock latency of HTTP requests.",
    labelnames=("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

INFERENCES_TOTAL = Counter(
    "factory_anomaly_inferences_total",
    "Number of /infer calls that produced a result, by model_version + anomaly verdict.",
    labelnames=("model_version", "is_anomaly"),
)

INFERENCE_DURATION = Histogram(
    "factory_anomaly_inference_duration_seconds",
    "Time spent inside /infer, excluding network and DB round-trip.",
    labelnames=("model_version",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

INFERENCE_SCORE = Histogram(
    "factory_anomaly_inference_score",
    "Distribution of anomaly scores returned by /infer (higher = more anomalous).",
    labelnames=("model_version",),
    buckets=(-0.5, -0.25, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0),
)

INGEST_ROWS_TOTAL = Counter(
    "factory_anomaly_ingest_rows_total",
    "Total sensor rows accepted by /ingest.",
)

MODEL_LOADED = Gauge(
    "factory_anomaly_model_loaded",
    "1 if a model artifact is loaded into this process, else 0.",
    labelnames=("model_version",),
)


def status_class(code: int) -> str:
    """Bucket an HTTP status code into a label-friendly class."""
    if 200 <= code < 300:
        return "2xx"
    if 300 <= code < 400:
        return "3xx"
    if 400 <= code < 500:
        return "4xx"
    if 500 <= code < 600:
        return "5xx"
    return "other"


def metrics_response(registry: CollectorRegistry | None = None) -> Response:
    """Render the Prometheus exposition format for the given registry."""
    payload = generate_latest(registry or DEFAULT_REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
