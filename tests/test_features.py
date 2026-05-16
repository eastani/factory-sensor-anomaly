"""Tests for rolling-window feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from factory_anomaly.ml.features import FEATURE_NAMES, make_rolling_features


def test_output_shape_drops_initial_partial_windows() -> None:
    values = np.arange(10, dtype=float)
    features = make_rolling_features(values, window=3)
    # 10 - 3 + 1 = 8 complete windows.
    assert features.shape == (8, len(FEATURE_NAMES))
    assert list(features.columns) == list(FEATURE_NAMES)


def test_known_values_for_simple_window() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    features = make_rolling_features(values, window=3)
    # First complete window is [1, 2, 3] -> mean=2, std=sqrt(2/3), min=1, max=3, p2p=2.
    first = features.iloc[0]
    assert first["mean"] == pytest.approx(2.0)
    assert first["std"] == pytest.approx(np.sqrt(2 / 3))
    assert first["min"] == pytest.approx(1.0)
    assert first["max"] == pytest.approx(3.0)
    assert first["p2p"] == pytest.approx(2.0)


def test_spike_lifts_max_and_p2p_in_overlapping_windows() -> None:
    values = np.zeros(20)
    values[10] = 100.0
    features = make_rolling_features(values, window=5)
    # The spike sits inside 5 different windows (indices ending at 10..14).
    flagged = features["p2p"] >= 100.0
    assert flagged.sum() == 5


def test_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        make_rolling_features(np.zeros((10, 2)), window=3)


def test_rejects_window_too_large() -> None:
    with pytest.raises(ValueError, match="at least window"):
        make_rolling_features(np.zeros(5), window=10)


def test_rejects_window_le_one() -> None:
    with pytest.raises(ValueError, match="window must be > 1"):
        make_rolling_features(np.zeros(5), window=1)
