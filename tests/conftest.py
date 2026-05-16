"""Pytest fixtures.

Integration tests spin up a real Postgres via testcontainers. The fixtures
here run Alembic migrations against that container so the test schema is
*always* the one a future operator will see, not a hand-rolled subset.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_url() -> Generator[str, None, None]:
    """Start a throwaway Postgres container for the entire test session."""
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as pg:
        yield pg.get_connection_url()


@pytest.fixture(scope="session")
def migrated_engine(postgres_url: str) -> Engine:
    """Engine pointing at a fully-migrated database."""
    engine = create_engine(postgres_url, future=True)

    cfg = Config("alembic.ini")
    cfg.set_main_option("script_location", "alembic")
    cfg.set_main_option("sqlalchemy.url", postgres_url)
    command.upgrade(cfg, "head")

    return engine


@pytest.fixture
def db_session(migrated_engine: Engine) -> Generator[Session, None, None]:
    """Per-test session wrapped in a rollback so tests stay isolated."""
    connection = migrated_engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection, autoflush=False, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
