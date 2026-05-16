"""Demo-data helper.

Generates a short window of synthetic readings, POSTs them to the API, then
fires one inference call. Used by the dashboard's "Generate demo data"
button so the rest of the UI has something to render.

Lives in importable code (not the Streamlit script) so it can be unit-tested.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from factory_anomaly.dashboard.client import ApiClient
from factory_anomaly.data import (
    inject_spikes,
    make_sine_wave,
)

DEMO_SENSOR_NAME = "sensor_00"


def _readings_payload(values: Iterable[float], *, sensor_name: str) -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    return [
        {
            "timestamp": (now + timedelta(seconds=i)).isoformat(),
            "sensor_name": sensor_name,
            "value": float(v),
        }
        for i, v in enumerate(values)
    ]


def generate_and_send(
    client: ApiClient,
    machine_id: str,
    *,
    n_samples: int = 60,
    with_spikes: bool = True,
    window_size: int = 20,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate one synthetic batch, ingest it, then run a single inference.

    Returns a small summary dict so the caller (UI) can render feedback.
    """
    base_seed = seed if seed is not None else int(datetime.now(UTC).timestamp())
    sig = make_sine_wave(n_samples=n_samples, period=20, amplitude=1.0, noise=0.1, seed=base_seed)
    if with_spikes:
        sig = inject_spikes(sig, count=max(1, n_samples // 15), magnitude=4.0, seed=base_seed + 1)

    payload = _readings_payload(sig.values.tolist(), sensor_name=DEMO_SENSOR_NAME)
    ingest_response = client.ingest(machine_id, payload)
    infer_response = client.infer(machine_id, DEMO_SENSOR_NAME, window_size=window_size)

    return {
        "machine_id": machine_id,
        "ingested": ingest_response["inserted"],
        "score": infer_response["score"],
        "is_anomaly": infer_response["is_anomaly"],
        "model_version": infer_response["model_version"],
    }
