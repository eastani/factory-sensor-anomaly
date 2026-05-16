"""Periodic synthetic-data ingester (see ADR-0004).

Generates a small batch of factory-pump-style readings every N seconds and
POSTs them to the API. Designed to be restartable: one missed tick is fine.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from factory_anomaly.dashboard.client import ApiClient, ApiClientError
from factory_anomaly.data import inject_spikes, make_sine_wave
from factory_anomaly.logging_config import configure_logging, get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class IngesterConfig:
    machine_id: str = "pump-001"
    sensor_name: str = "sensor_00"
    batch_size: int = 30
    spike_probability: float = 0.2
    interval_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> IngesterConfig:
        return cls(
            machine_id=os.environ.get("INGESTER_MACHINE_ID", "pump-001"),
            sensor_name=os.environ.get("INGESTER_SENSOR_NAME", "sensor_00"),
            batch_size=int(os.environ.get("INGESTER_BATCH_SIZE", "30")),
            spike_probability=float(os.environ.get("INGESTER_SPIKE_PROBABILITY", "0.2")),
            interval_seconds=float(os.environ.get("INGESTER_INTERVAL_SECONDS", "10")),
        )


def _payload(values: list[float], sensor_name: str) -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    return [
        {
            "timestamp": (now + timedelta(milliseconds=i * 100)).isoformat(),
            "sensor_name": sensor_name,
            "value": float(v),
        }
        for i, v in enumerate(values)
    ]


def run_one_tick(client: ApiClient, cfg: IngesterConfig, *, tick: int, seed: int) -> bool:
    """Generate one batch and POST it. Returns True on success."""
    sig = make_sine_wave(
        n_samples=cfg.batch_size,
        period=15,
        amplitude=1.0,
        noise=0.1,
        seed=seed,
    )
    if cfg.spike_probability > 0 and (seed % 100) / 100 < cfg.spike_probability:
        sig = inject_spikes(sig, count=2, magnitude=4.0, seed=seed + 1)

    try:
        resp = client.ingest(cfg.machine_id, _payload(sig.values.tolist(), cfg.sensor_name))
        log.info(
            "ingester.posted",
            tick=tick,
            inserted=resp["inserted"],
            spiked=sig.n_anomalies > 0,
        )
        return True
    except ApiClientError as exc:
        log.warning("ingester.api_error", tick=tick, error=str(exc))
        return False


def main() -> None:  # pragma: no cover — exercised end-to-end via docker compose
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    cfg = IngesterConfig.from_env()
    api_base_url = os.environ.get("API_BASE_URL", "http://api:8000")

    log.info(
        "ingester.start",
        api_base_url=api_base_url,
        machine_id=cfg.machine_id,
        sensor_name=cfg.sensor_name,
        interval=cfg.interval_seconds,
        batch_size=cfg.batch_size,
    )

    tick = 0
    with ApiClient(api_base_url, timeout=10.0) as client:
        while True:
            tick += 1
            seed = int(datetime.now(UTC).timestamp() * 1_000) % (2**31 - 1)
            run_one_tick(client, cfg, tick=tick, seed=seed)
            time.sleep(cfg.interval_seconds)


if __name__ == "__main__":  # pragma: no cover
    main()
