"""FastAPI dependency providers.

Kept tiny on purpose: each provider returns one resource and one resource only,
so tests can override exactly the boundary they care about (``get_session``
swapped for a testcontainers session, ``get_detector`` swapped for a stub).
"""

from __future__ import annotations

from collections.abc import Generator

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from factory_anomaly.ml import AnomalyDetector


def get_session(request: Request) -> Generator[Session, None, None]:
    """Yield a DB session for the lifetime of one request."""
    session_factory = request.app.state.session_factory
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_detector(request: Request) -> AnomalyDetector:
    """Return the process-wide AnomalyDetector or 503 if it was not loaded."""
    detector: AnomalyDetector | None = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model artifact not loaded",
        )
    return detector
