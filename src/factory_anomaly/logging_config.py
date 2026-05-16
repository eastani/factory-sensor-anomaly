"""Structured JSON logging via structlog.

Configures both stdlib ``logging`` (so uvicorn / sqlalchemy output joins the
same stream) and structlog's bound logger. Calling ``configure_logging`` is
idempotent — safe to invoke from app startup and from individual tests.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Set up JSON structured logging at the given level."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Convenience accessor for module-level loggers."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
