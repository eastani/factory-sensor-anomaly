"""Initial schema: sensor_readings + anomaly_results.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-05-16
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sensor_readings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("machine_id", sa.String(length=64), nullable=False),
        sa.Column("sensor_name", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_sensor_readings_machine_ts",
        "sensor_readings",
        ["machine_id", "timestamp"],
    )
    op.create_index("ix_sensor_readings_ts", "sensor_readings", ["timestamp"])

    op.create_table(
        "anomaly_results",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("machine_id", sa.String(length=64), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_anomaly_results_machine_ts",
        "anomaly_results",
        ["machine_id", "timestamp"],
    )
    op.create_index(
        "ix_anomaly_results_model",
        "anomaly_results",
        ["model_name", "model_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_anomaly_results_model", table_name="anomaly_results")
    op.drop_index("ix_anomaly_results_machine_ts", table_name="anomaly_results")
    op.drop_table("anomaly_results")
    op.drop_index("ix_sensor_readings_ts", table_name="sensor_readings")
    op.drop_index("ix_sensor_readings_machine_ts", table_name="sensor_readings")
    op.drop_table("sensor_readings")
