"""Synthetic factory-sensor signals for development, demos, and unit tests.

Real datasets (Kaggle Pump, SKAB) require download and friction. These
generators give the rest of the pipeline a zero-setup substrate: every
function is deterministic given a seed, and every signal carries its own
ground-truth anomaly mask so downstream evaluation does not have to guess.

The vocabulary mirrors how field engineers describe failures:

- **Spike** — a single-sample outlier (sensor glitch, momentary impulse).
- **Trend** — a gradual drift (bearing wear, calibration drift).
- **Level shift** — an abrupt step (load change, valve toggle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class SyntheticSignal:
    """A 1-D signal plus the anomaly ground truth that produced it.

    ``values`` and ``anomaly_mask`` are parallel arrays; ``timestamps`` is a
    convenience for plotting and DB ingestion. All arrays have the same length.
    """

    values: np.ndarray
    timestamps: np.ndarray
    anomaly_mask: np.ndarray
    description: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.values)
        if not (len(self.timestamps) == len(self.anomaly_mask) == n):
            raise ValueError(
                "values, timestamps, anomaly_mask must have identical length; "
                f"got {len(self.values)}, {len(self.timestamps)}, {len(self.anomaly_mask)}"
            )
        if self.anomaly_mask.dtype != np.bool_:
            raise TypeError(f"anomaly_mask must be bool, got {self.anomaly_mask.dtype}")

    @property
    def n_samples(self) -> int:
        return len(self.values)

    @property
    def n_anomalies(self) -> int:
        return int(self.anomaly_mask.sum())


def make_sine_wave(
    n_samples: int = 1_000,
    *,
    period: int = 50,
    amplitude: float = 1.0,
    noise: float = 0.05,
    start: datetime | None = None,
    sample_interval: timedelta = timedelta(seconds=1),
    seed: int = 0,
) -> SyntheticSignal:
    """Generate a base normal signal: pure sine + Gaussian noise, no anomalies.

    ``period`` is in samples, not seconds — keeps the function unit-free.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    values = amplitude * np.sin(2 * np.pi * t / period) + rng.normal(0, noise, n_samples)

    # numpy.datetime64 does not carry timezone info — drop tz to silence the
    # UserWarning, callers can re-attach UTC at the application boundary.
    base = (start or datetime.now(UTC)).replace(tzinfo=None)
    timestamps = np.array(
        [base + i * sample_interval for i in range(n_samples)], dtype="datetime64[ns]"
    )

    return SyntheticSignal(
        values=values,
        timestamps=timestamps,
        anomaly_mask=np.zeros(n_samples, dtype=bool),
        description=f"sine(period={period}, amp={amplitude}, noise={noise})",
        metadata={"period": period, "amplitude": amplitude, "noise": noise, "seed": seed},
    )


def inject_spikes(
    signal: SyntheticSignal,
    *,
    count: int = 5,
    magnitude: float = 5.0,
    seed: int = 1,
) -> SyntheticSignal:
    """Add ``count`` point spikes of size ``magnitude`` at random positions.

    Spikes are placed only where the existing mask is False so anomaly counts
    stay accurate if you chain injectors.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    rng = np.random.default_rng(seed)
    available = np.flatnonzero(~signal.anomaly_mask)
    if len(available) < count:
        raise ValueError(f"only {len(available)} non-anomaly samples available, need {count}")

    indices = rng.choice(available, size=count, replace=False)
    signs = rng.choice([-1.0, 1.0], size=count)

    new_values = signal.values.copy()
    new_values[indices] += signs * magnitude

    new_mask = signal.anomaly_mask.copy()
    new_mask[indices] = True

    return SyntheticSignal(
        values=new_values,
        timestamps=signal.timestamps,
        anomaly_mask=new_mask,
        description=f"{signal.description} + {count}×spike(mag={magnitude})",
        metadata={**signal.metadata, "spike_indices": indices.tolist()},
    )


def inject_trend(
    signal: SyntheticSignal,
    *,
    start_index: int,
    slope: float = 0.01,
) -> SyntheticSignal:
    """Add a linear drift starting at ``start_index`` and continuing to the end."""
    n = signal.n_samples
    if not 0 <= start_index < n:
        raise ValueError(f"start_index {start_index} out of range [0, {n})")

    drift = np.zeros(n)
    drift[start_index:] = slope * np.arange(n - start_index)

    new_mask = signal.anomaly_mask.copy()
    new_mask[start_index:] = True

    return SyntheticSignal(
        values=signal.values + drift,
        timestamps=signal.timestamps,
        anomaly_mask=new_mask,
        description=f"{signal.description} + trend(start={start_index}, slope={slope})",
        metadata={**signal.metadata, "trend_start": start_index, "trend_slope": slope},
    )


def inject_level_shift(
    signal: SyntheticSignal,
    *,
    start_index: int,
    magnitude: float = 2.0,
) -> SyntheticSignal:
    """Add an abrupt step change of ``magnitude`` starting at ``start_index``."""
    n = signal.n_samples
    if not 0 <= start_index < n:
        raise ValueError(f"start_index {start_index} out of range [0, {n})")

    new_values = signal.values.copy()
    new_values[start_index:] += magnitude

    new_mask = signal.anomaly_mask.copy()
    new_mask[start_index:] = True

    return SyntheticSignal(
        values=new_values,
        timestamps=signal.timestamps,
        anomaly_mask=new_mask,
        description=f"{signal.description} + level_shift(start={start_index}, mag={magnitude})",
        metadata={
            **signal.metadata,
            "level_shift_start": start_index,
            "level_shift_magnitude": magnitude,
        },
    )


def make_example_dataset(seed: int = 42) -> SyntheticSignal:
    """Compose a canonical example signal used across tests and demos.

    1000 samples of a periodic base, then a handful of spikes and one drift
    near the end. The proportions are tuned so a baseline Isolation Forest
    catches roughly the right anomalies without parameter sweeping.
    """
    base = make_sine_wave(n_samples=1_000, period=50, amplitude=1.0, noise=0.1, seed=seed)
    with_spikes = inject_spikes(base, count=8, magnitude=4.0, seed=seed + 1)
    return inject_trend(with_spikes, start_index=850, slope=0.02)
