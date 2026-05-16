"""SQLAlchemy engine and session factory helpers.

Engines are *expensive* (they hold a connection pool); session makers are
*cheap*. Call ``create_engine_from_settings`` once at process start, and
``create_session_factory(engine)`` to get a ``sessionmaker`` for that engine.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from factory_anomaly.config import DatabaseSettings, get_database_settings


def create_engine_from_settings(
    settings: DatabaseSettings | None = None,
    *,
    echo: bool = False,
) -> Engine:
    """Build a synchronous SQLAlchemy ``Engine`` from settings.

    ``pool_pre_ping`` guards against stale connections after Postgres restarts
    (cheap ``SELECT 1`` round-trip on checkout).
    """
    cfg = settings or get_database_settings()
    return create_engine(cfg.url, echo=echo, pool_pre_ping=True, future=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a ``sessionmaker`` bound to the given engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
