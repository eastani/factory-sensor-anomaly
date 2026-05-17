"""Tests for the STL-residual anomaly detector."""

from __future__ import annotations

import numpy as np
import pytest

from factory_anomaly.data import inject_spikes, make_sine_wave
from factory_anomaly.ml import StlAnomalyDetector


@pytest.fixture
def cyclic_signal() -> np.ndarray:
    """A clean sine wave long enough to STL-decompose at period=20."""
    return make_sine_wave(n_samples=400, period=20, amplitude=1.0, noise=0.05, seed=0).values


def test_rejects_invalid_period() -> None:
    with pytest.raises(ValueError, match="period must be"):
        StlAnomalyDetector(model_version="v1", period=1)


def test_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window must be"):
        StlAnomalyDetector(model_version="v1", period=20, window=1)


def test_residuals_remove_dominant_seasonality(cyclic_signal: np.ndarray) -> None:
    """A pure sine wave's residual should be much smaller than the original."""
    det = StlAnomalyDetector(model_version="v1", period=20, window=10, robust=False)
    residuals = det._residuals(cyclic_signal)
    assert residuals.shape == cyclic_signal.shape
    # Residual std must be substantially smaller than raw std for a periodic input.
    assert residuals.std() < 0.5 * cyclic_signal.std()


def test_fit_then_score_returns_per_window_floats(cyclic_signal: np.ndarray) -> None:
    det = StlAnomalyDetector(model_version="v1", period=20, window=10, robust=False)
    det.fit(cyclic_signal)
    scores = det.score(cyclic_signal)
    expected_rows = len(cyclic_signal) - 10 + 1
    assert scores.shape == (expected_rows,)
    assert scores.dtype == np.float64


def test_metadata_carries_model_version(cyclic_signal: np.ndarray) -> None:
    det = StlAnomalyDetector(model_version="stl-test", period=20, window=10, robust=False)
    det.fit(cyclic_signal)
    assert det.metadata.model_version == "stl-test"
    assert det.is_fitted is True


def test_hyperparameters_surfaces_stl_params() -> None:
    det = StlAnomalyDetector(model_version="v1", period=30, window=15, robust=False)
    params = det.hyperparameters()
    assert params["period"] == 30
    assert params["window"] == 15
    assert params["robust"] is False
    # IF params come from the inner estimator.
    assert params["n_estimators"] == 100


def test_stl_lifts_anomaly_score_for_injected_spikes(cyclic_signal: np.ndarray) -> None:
    """The most important behavioural test: STL+IF should rank a spiky run higher.

    Train on clean cyclic data, then score (a) the same clean data and
    (b) the same data with a spike injected near the end. The spiky
    window must score strictly higher than the corresponding clean window.
    """
    det = StlAnomalyDetector(model_version="v1", period=20, window=10, robust=False)
    det.fit(cyclic_signal)
    clean_scores = det.score(cyclic_signal)

    # Inject a large spike at index 350 and re-score.
    spike_values = cyclic_signal.copy()
    spike_values[350] += 10.0
    spike_scores = det.score(spike_values)

    # Right-edge alignment means the window ending at index 350 is at
    # position (350 - window + 1).
    spike_window_idx = 350 - 10 + 1
    assert spike_scores[spike_window_idx] > clean_scores[spike_window_idx]


def test_rejects_2d_input() -> None:
    det = StlAnomalyDetector(model_version="v1", period=20, window=10, robust=False)
    with pytest.raises(ValueError, match="1-D"):
        det._residuals(np.zeros((100, 2)))


@pytest.mark.slow
def test_fits_on_signal_with_real_spikes() -> None:
    """Integration-ish: end-to-end fit + score on a sine + injected spikes."""
    sig = make_sine_wave(n_samples=600, period=30, amplitude=1.0, noise=0.1, seed=42)
    sig = inject_spikes(sig, count=5, magnitude=3.5, seed=43)

    det = StlAnomalyDetector(model_version="v1", period=30, window=15, robust=True)
    det.fit(sig.values)
    scores = det.score(sig.values)
    # All scores finite, no NaN/Inf bleeding through from STL.
    assert np.isfinite(scores).all()
