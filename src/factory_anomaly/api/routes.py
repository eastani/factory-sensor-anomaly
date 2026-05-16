"""HTTP routes for the anomaly detection service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import desc, select, text
from sqlalchemy.orm import Session

from factory_anomaly.api.dependencies import get_detector, get_session
from factory_anomaly.api.schemas import (
    AnomalyOut,
    HealthResponse,
    InferRequest,
    IngestRequest,
    IngestResponse,
    ReadingOut,
)
from factory_anomaly.config import get_api_settings
from factory_anomaly.db import AnomalyResult, SensorReading
from factory_anomaly.logging_config import get_logger
from factory_anomaly.ml import AnomalyDetector
from factory_anomaly.ml.features import make_rolling_features

router = APIRouter()
log = get_logger(__name__)

SessionDep = Annotated[Session, Depends(get_session)]
DetectorDep = Annotated[AnomalyDetector, Depends(get_detector)]


@router.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz(
    request: Request,
    session: SessionDep,
) -> HealthResponse:
    """Liveness + readiness in one endpoint.

    Reports OK status if and only if every dependency is healthy; otherwise
    returns 200 with ``status="degraded"`` so monitors can distinguish
    "service alive but useless" from "service unreachable".
    """
    db_status = "ok"
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover — exercised in integration only
        db_status = f"error: {exc.__class__.__name__}"

    detector: AnomalyDetector | None = getattr(request.app.state, "detector", None)
    overall_ok = db_status == "ok" and detector is not None
    return HealthResponse(
        status="ok" if overall_ok else "degraded",
        db=db_status,
        model="loaded" if detector is not None else "unavailable",
        model_version=detector.metadata.model_version if detector else None,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingest"],
)
def ingest_readings(
    payload: IngestRequest,
    session: SessionDep,
) -> IngestResponse:
    rows = [
        SensorReading(
            timestamp=r.timestamp,
            machine_id=payload.machine_id,
            sensor_name=r.sensor_name,
            value=r.value,
        )
        for r in payload.readings
    ]
    session.add_all(rows)
    session.flush()
    log.info("ingest", machine_id=payload.machine_id, count=len(rows), first_ts=rows[0].timestamp)
    return IngestResponse(inserted=len(rows))


@router.get(
    "/readings/{machine_id}",
    response_model=list[ReadingOut],
    tags=["read"],
)
def list_readings(
    machine_id: str,
    session: SessionDep,
    sensor_name: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=10_000),
) -> list[ReadingOut]:
    stmt = (
        select(SensorReading)
        .where(SensorReading.machine_id == machine_id)
        .order_by(desc(SensorReading.timestamp))
        .limit(limit)
    )
    if sensor_name:
        stmt = stmt.where(SensorReading.sensor_name == sensor_name)
    rows = session.execute(stmt).scalars().all()
    return [ReadingOut.model_validate(r) for r in rows]


@router.get(
    "/anomalies/{machine_id}",
    response_model=list[AnomalyOut],
    tags=["read"],
)
def list_anomalies(
    machine_id: str,
    session: SessionDep,
    limit: int = Query(default=100, ge=1, le=10_000),
    only_anomalies: bool = Query(default=False),
) -> list[AnomalyOut]:
    stmt = (
        select(AnomalyResult)
        .where(AnomalyResult.machine_id == machine_id)
        .order_by(desc(AnomalyResult.timestamp))
        .limit(limit)
    )
    if only_anomalies:
        stmt = stmt.where(AnomalyResult.is_anomaly.is_(True))
    rows = session.execute(stmt).scalars().all()
    return [AnomalyOut.model_validate(r) for r in rows]


@router.post(
    "/infer",
    response_model=AnomalyOut,
    status_code=status.HTTP_201_CREATED,
    tags=["infer"],
)
def infer(
    payload: InferRequest,
    session: SessionDep,
    detector: DetectorDep,
) -> AnomalyOut:
    window = payload.window_size or get_api_settings().window_size

    stmt = (
        select(SensorReading)
        .where(
            SensorReading.machine_id == payload.machine_id,
            SensorReading.sensor_name == payload.sensor_name,
        )
        .order_by(desc(SensorReading.timestamp))
        .limit(window)
    )
    rows = list(session.execute(stmt).scalars().all())
    if len(rows) < window:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"need {window} readings for machine_id={payload.machine_id!r} "
                f"sensor_name={payload.sensor_name!r}, found {len(rows)}"
            ),
        )

    # ``rows`` is newest-first; reverse so timestamps and features are causal.
    rows = list(reversed(rows))
    values = np.asarray([r.value for r in rows], dtype=float)
    features = make_rolling_features(values, window=window).to_numpy()
    score = float(detector.score(features)[0])
    is_anomaly = bool(detector.predict(features)[0])

    result = AnomalyResult(
        timestamp=datetime.now(UTC),
        machine_id=payload.machine_id,
        window_start=rows[0].timestamp,
        window_end=rows[-1].timestamp,
        score=score,
        is_anomaly=is_anomaly,
        model_name=detector.metadata.model_name,
        model_version=detector.metadata.model_version,
    )
    session.add(result)
    session.flush()

    log.info(
        "infer",
        machine_id=payload.machine_id,
        sensor_name=payload.sensor_name,
        score=score,
        is_anomaly=is_anomaly,
        model_version=detector.metadata.model_version,
    )
    return AnomalyOut.model_validate(result)
