"""Pytest fixtures.

Integration tests spin up a real Postgres via testcontainers. The fixtures
here run Alembic migrations against that container so the test schema is
*always* the one a future operator will see, not a hand-rolled subset.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from urllib.parse import urlparse

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

from factory_anomaly.data import make_example_dataset
from factory_anomaly.ml import AnomalyDetector
from factory_anomaly.ml.features import make_rolling_features


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


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train and persist a baseline model once per session."""
    out = tmp_path_factory.mktemp("models") / "baseline.joblib"
    sig = make_example_dataset(seed=42)
    features = make_rolling_features(sig.values, window=20).to_numpy()
    detector = AnomalyDetector(model_version="test-v1")
    detector.fit(features)
    detector.save(out)
    return out


@pytest.fixture
def api_client(
    postgres_url: str,
    migrated_engine: Engine,
    trained_model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """A FastAPI TestClient wired to the test Postgres + a real trained model."""
    parsed = urlparse(postgres_url)
    monkeypatch.setenv("POSTGRES_USER", parsed.username or "")
    monkeypatch.setenv("POSTGRES_PASSWORD", parsed.password or "")
    monkeypatch.setenv("POSTGRES_HOST", parsed.hostname or "")
    monkeypatch.setenv("POSTGRES_PORT", str(parsed.port or 5432))
    monkeypatch.setenv("POSTGRES_DB", parsed.path.lstrip("/"))
    monkeypatch.setenv("API_MODEL_PATH", str(trained_model_path))

    from factory_anomaly.config import get_api_settings, get_database_settings

    get_api_settings.cache_clear()
    get_database_settings.cache_clear()

    from factory_anomaly.api import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client

    get_api_settings.cache_clear()
    get_database_settings.cache_clear()
