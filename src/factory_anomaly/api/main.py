"""FastAPI application factory.

The application is composed via ``create_app`` rather than a module-level
``app =`` so that:

1. Tests can spin up isolated instances with different settings without
   monkey-patching globals.
2. Cloud platforms that expect ``app:create_app`` (factory pattern) or
   ``app:app`` (direct) both work cleanly.

The lifespan handler is where the *expensive* one-time work lives: opening
the DB connection pool and loading the model artifact. Per ADR-0002, the
model must be loaded once at process start, never per request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.responses import Response

from factory_anomaly.api.routes import router
from factory_anomaly.config import get_api_settings, get_database_settings
from factory_anomaly.db.session import create_engine_from_settings, create_session_factory
from factory_anomaly.logging_config import configure_logging, get_logger
from factory_anomaly.ml import AnomalyDetector, SklearnVersionMismatchError
from factory_anomaly.observability import (
    MODEL_LOADED,
    metrics_response,
    register_metrics_middleware,
)

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the engine + load the model on startup; tear down on shutdown."""
    api = get_api_settings()
    configure_logging(api.log_level)

    db = get_database_settings()
    engine = create_engine_from_settings(db)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    log.info("startup.db_engine_ready", url=db.url.split("@")[-1])

    try:
        detector = AnomalyDetector.load(api.model_path)
        app.state.detector = detector
        MODEL_LOADED.labels(model_version=detector.metadata.model_version).set(1)
        log.info(
            "startup.model_loaded",
            path=str(api.model_path),
            version=detector.metadata.model_version,
            sklearn_version=detector.metadata.sklearn_version,
        )
    except (FileNotFoundError, SklearnVersionMismatchError) as exc:
        app.state.detector = None
        MODEL_LOADED.labels(model_version="none").set(0)
        log.warning(
            "startup.model_unavailable",
            path=str(api.model_path),
            reason=exc.__class__.__name__,
            message=str(exc),
        )

    try:
        yield
    finally:
        engine.dispose()
        log.info("shutdown.db_engine_disposed")


def create_app() -> FastAPI:
    """Build a fresh FastAPI app instance."""
    app = FastAPI(
        title="factory-sensor-anomaly",
        description=(
            "Real-time anomaly detection for factory sensor streams. "
            "Ingest readings, run unsupervised inference, expose results."
        ),
        version="0.0.1",
        lifespan=lifespan,
    )
    app.include_router(router)
    register_metrics_middleware(app)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return metrics_response()

    return app


# Module-level ASGI handle for `uvicorn factory_anomaly.api.main:app`.
app = create_app()
