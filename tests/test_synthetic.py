"""Tests for the synthetic data generators."""

from __future__ import annotations

import numpy as np
import pytest

from factory_anomaly.data import (
    SyntheticSignal,
    inject_level_shift,
    inject_spikes,
    inject_trend,
    make_example_dataset,
    make_sine_wave,
)


def test_sine_wave_shape_and_no_anomalies() -> None:
    sig = make_sine_wave(n_samples=200, period=20, amplitude=2.0, noise=0.0, seed=0)
    assert sig.n_samples == 200
    assert sig.n_anomalies == 0
    assert sig.values.shape == (200,)
    assert sig.timestamps.shape == (200,)
    # Without noise the signal is exactly amplitude * sin — bounds should hold.
    assert sig.values.min() >= -2.0 - 1e-9
    assert sig.values.max() <= 2.0 + 1e-9


def test_sine_wave_is_deterministic_given_seed() -> None:
    a = make_sine_wave(n_samples=100, seed=7)
    b = make_sine_wave(n_samples=100, seed=7)
    np.testing.assert_array_equal(a.values, b.values)


def test_invalid_n_samples_rejected() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        make_sine_wave(n_samples=0)


def test_inject_spikes_marks_anomalies_and_shifts_values() -> None:
    base = make_sine_wave(n_samples=500, noise=0.0, seed=1)
    out = inject_spikes(base, count=10, magnitude=5.0, seed=1)
    assert out.n_anomalies == 10
    # The points that became anomalies must differ from the base.
    diff_indices = np.flatnonzero(out.values != base.values)
    assert len(diff_indices) == 10
    np.testing.assert_array_equal(np.sort(diff_indices), np.sort(np.flatnonzero(out.anomaly_mask)))


def test_inject_spikes_refuses_to_overflow_available_slots() -> None:
    base = make_sine_wave(n_samples=20, seed=0)
    almost_full = inject_spikes(base, count=18, magnitude=1.0, seed=0)
    with pytest.raises(ValueError, match="non-anomaly samples available"):
        inject_spikes(almost_full, count=5, magnitude=1.0, seed=1)


def test_inject_trend_changes_tail_and_marks_mask() -> None:
    base = make_sine_wave(n_samples=100, noise=0.0, seed=0)
    out = inject_trend(base, start_index=80, slope=0.5)
    assert out.anomaly_mask[:80].sum() == 0
    assert out.anomaly_mask[80:].all()
    np.testing.assert_array_equal(out.values[:80], base.values[:80])
    assert out.values[-1] > base.values[-1]


def test_inject_level_shift_is_constant_offset() -> None:
    base = make_sine_wave(n_samples=100, noise=0.0, seed=0)
    out = inject_level_shift(base, start_index=50, magnitude=3.0)
    np.testing.assert_allclose(out.values[50:] - base.values[50:], 3.0)
    assert out.anomaly_mask[50:].all()


def test_make_example_dataset_has_expected_composition() -> None:
    sig = make_example_dataset()
    assert sig.n_samples == 1_000
    # 8 spikes + 150-sample trend = at least 8 anomaly points.
    assert sig.n_anomalies >= 8
    # Trend region must be fully flagged.
    assert sig.anomaly_mask[850:].all()


def test_synthetic_signal_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="identical length"):
        SyntheticSignal(
            values=np.zeros(10),
            timestamps=np.zeros(10, dtype="datetime64[ns]"),
            anomaly_mask=np.zeros(5, dtype=bool),
        )


def test_synthetic_signal_rejects_non_bool_mask() -> None:
    with pytest.raises(TypeError, match="bool"):
        SyntheticSignal(
            values=np.zeros(5),
            timestamps=np.zeros(5, dtype="datetime64[ns]"),
            anomaly_mask=np.zeros(5, dtype=np.int8),
        )
