"""HTTP middleware that records request count + latency to Prometheus.

The route *template* (``/readings/{machine_id}``) is used as a label rather
than the resolved path, so cardinality stays bounded under arbitrary machine
ids.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from factory_anomaly.observability.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    status_class,
)


def _route_template(request: Request) -> str:
    """Resolve the matched route template, or fall back to the raw URL path."""
    route = request.scope.get("route")
    path: str | None = getattr(route, "path", None) if route is not None else None
    return path or request.url.path


def register_metrics_middleware(app: FastAPI) -> None:
    """Attach the request-counter / latency-histogram middleware to ``app``."""

    @app.middleware("http")
    async def metrics_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        route = _route_template(request)
        if route == "/metrics":
            # Don't count scraping itself — turns the metric into a fixed point.
            return response

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            route=route,
            status_class=status_class(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION.labels(method=request.method, route=route).observe(elapsed)
        return response
