"""Data layer — synthetic generators (Phase 1.2) and real-dataset loaders (later)."""

from factory_anomaly.data.synthetic import (
    SyntheticSignal,
    inject_level_shift,
    inject_spikes,
    inject_trend,
    make_example_dataset,
    make_sine_wave,
)

__all__ = [
    "SyntheticSignal",
    "inject_level_shift",
    "inject_spikes",
    "inject_trend",
    "make_example_dataset",
    "make_sine_wave",
]
