"""Integration tests for the DB layer.

These tests spin up a real Postgres via testcontainers (see ``conftest.py``)
and run the production Alembic migrations against it. They are marked
``integration`` because they need Docker to be available.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import Session

from factory_anomaly.db import AnomalyResult, SensorReading

pytestmark = pytest.mark.integration


def test_migrations_create_expected_tables_and_indexes(db_session: Session) -> None:
    """Smoke check: every table and index the ORM declares exists in the DB."""
    bind = db_session.bind
    assert bind is not None
    inspector: Inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    assert {"sensor_readings", "anomaly_results"}.issubset(tables)
    assert "alembic_version" in tables  # migrations were actually applied

    sensor_indexes = {idx["name"] for idx in inspector.get_indexes("sensor_readings")}
    assert {"ix_sensor_readings_machine_ts", "ix_sensor_readings_ts"} <= sensor_indexes

    anomaly_indexes = {idx["name"] for idx in inspector.get_indexes("anomaly_results")}
    assert {"ix_anomaly_results_machine_ts", "ix_anomaly_results_model"} <= anomaly_indexes


def test_sensor_reading_round_trip(db_session: Session) -> None:
    """Insert a reading, read it back, and confirm timezone + value preserved."""
    now = datetime.now(UTC)
    reading = SensorReading(
        timestamp=now,
        machine_id="pump-001",
        sensor_name="sensor_00",
        value=42.5,
    )
    db_session.add(reading)
    db_session.flush()

    fetched = db_session.scalar(select(SensorReading).where(SensorReading.id == reading.id))
    assert fetched is not None
    assert fetched.machine_id == "pump-001"
    assert fetched.value == pytest.approx(42.5)
    assert fetched.timestamp.tzinfo is not None
    assert fetched.ingested_at.tzinfo is not None


def test_anomaly_result_round_trip(db_session: Session) -> None:
    """Insert a result and verify model metadata + score round-trip cleanly."""
    now = datetime.now(UTC)
    result = AnomalyResult(
        timestamp=now,
        machine_id="pump-001",
        window_start=now - timedelta(minutes=5),
        window_end=now,
        score=-0.31,
        is_anomaly=True,
        model_name="isolation_forest",
        model_version="2026-05-16-v1",
    )
    db_session.add(result)
    db_session.flush()

    fetched = db_session.scalar(select(AnomalyResult).where(AnomalyResult.id == result.id))
    assert fetched is not None
    assert fetched.is_anomaly is True
    assert fetched.score == pytest.approx(-0.31)
    assert fetched.model_version == "2026-05-16-v1"
