"""Periodic inference trigger (see ADR-0004).

Polls ``POST /infer`` every N seconds. Expected transient errors during
startup (model still loading, not enough readings yet) are logged at WARN
but never fatal.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from factory_anomaly.dashboard.client import ApiClient, ApiClientError
from factory_anomaly.logging_config import configure_logging, get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ScorerConfig:
    machine_id: str = "pump-001"
    sensor_name: str = "sensor_00"
    window_size: int | None = None
    interval_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> ScorerConfig:
        window_size_env = os.environ.get("SCORER_WINDOW_SIZE")
        return cls(
            machine_id=os.environ.get("SCORER_MACHINE_ID", "pump-001"),
            sensor_name=os.environ.get("SCORER_SENSOR_NAME", "sensor_00"),
            window_size=int(window_size_env) if window_size_env else None,
            interval_seconds=float(os.environ.get("SCORER_INTERVAL_SECONDS", "5")),
        )


def run_one_tick(client: ApiClient, cfg: ScorerConfig, *, tick: int) -> bool:
    """Call ``/infer`` once. Returns True on success, False on any error."""
    try:
        resp = client.infer(cfg.machine_id, cfg.sensor_name, window_size=cfg.window_size)
        log.info(
            "scorer.inferred",
            tick=tick,
            score=resp["score"],
            is_anomaly=resp["is_anomaly"],
            model_version=resp["model_version"],
        )
        return True
    except ApiClientError as exc:
        # 422 (not enough data yet) and 503 (model still loading) are both
        # expected during startup — they must not crash the worker.
        if exc.status_code in (422, 503):
            log.warning("scorer.transient", tick=tick, status_code=exc.status_code, error=str(exc))
        else:
            log.error("scorer.api_error", tick=tick, error=str(exc))
        return False


def main() -> None:  # pragma: no cover — exercised via docker compose
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    cfg = ScorerConfig.from_env()
    api_base_url = os.environ.get("API_BASE_URL", "http://api:8000")

    log.info(
        "scorer.start",
        api_base_url=api_base_url,
        machine_id=cfg.machine_id,
        sensor_name=cfg.sensor_name,
        interval=cfg.interval_seconds,
        window_size=cfg.window_size,
    )

    tick = 0
    with ApiClient(api_base_url, timeout=10.0) as client:
        while True:
            tick += 1
            run_one_tick(client, cfg, tick=tick)
            time.sleep(cfg.interval_seconds)


if __name__ == "__main__":  # pragma: no cover
    main()
