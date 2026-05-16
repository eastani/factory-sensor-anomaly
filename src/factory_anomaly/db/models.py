"""SQLAlchemy 2.x ORM models.

Storage strategy is **long-format** (one row per sensor reading) rather than wide
(one column per sensor). This decision is intentional and documented in
ADR-0001: the same schema must hold both the Kaggle Pump dataset (52 sensors)
and SKAB (a different sensor set) without per-dataset migrations.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Index, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class SensorReading(Base):
    """A single sensor measurement.

    Long-format row: ``(timestamp, machine_id, sensor_name) -> value``. Combined
    with the ``(machine_id, timestamp DESC)`` index this gives fast
    "latest-N readings for asset X" queries that the dashboard needs.
    """

    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    machine_id: Mapped[str] = mapped_column(String(64), nullable=False)
    sensor_name: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_sensor_readings_machine_ts", "machine_id", "timestamp"),
        Index("ix_sensor_readings_ts", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"SensorReading(machine_id={self.machine_id!r}, "
            f"sensor={self.sensor_name!r}, ts={self.timestamp.isoformat()}, "
            f"value={self.value})"
        )


class AnomalyResult(Base):
    """A model's verdict on a window of sensor data.

    ``model_version`` is part of the row, not a separate dimension table, so we
    can compare old and new model behaviour by running both against the same
    data and keeping their results side-by-side.
    """

    __tablename__ = "anomaly_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    machine_id: Mapped[str] = mapped_column(String(64), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_anomaly_results_machine_ts", "machine_id", "timestamp"),
        Index("ix_anomaly_results_model", "model_name", "model_version"),
    )

    def __repr__(self) -> str:
        return (
            f"AnomalyResult(machine_id={self.machine_id!r}, "
            f"ts={self.timestamp.isoformat()}, score={self.score:.4f}, "
            f"is_anomaly={self.is_anomaly}, model={self.model_name}@{self.model_version})"
        )
