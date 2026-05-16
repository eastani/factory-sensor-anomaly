"""Pydantic v2 request / response schemas for the API.

Schemas are intentionally separate from the SQLAlchemy ORM models. The DB
models describe how data is *stored*; the schemas here describe the API
*contract*. They can evolve independently — e.g. internal index columns
should never leak to the wire.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' if every dependency is healthy.")
    db: str = Field(description="'ok' or an error string.")
    model: str = Field(description="'loaded' or 'unavailable'.")
    model_version: str | None = None


class ReadingIn(BaseModel):
    timestamp: datetime
    sensor_name: str = Field(min_length=1, max_length=64)
    value: float


class IngestRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=64)
    readings: list[ReadingIn] = Field(min_length=1)


class IngestResponse(BaseModel):
    inserted: int


class ReadingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    machine_id: str
    sensor_name: str
    value: float


class InferRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=64)
    sensor_name: str = Field(min_length=1, max_length=64)
    window_size: int | None = Field(default=None, ge=2)


class AnomalyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    machine_id: str
    window_start: datetime
    window_end: datetime
    score: float
    is_anomaly: bool
    model_name: str
    model_version: str
