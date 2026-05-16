"""Tests for the drift primitives."""

from __future__ import annotations

import numpy as np
import pytest

from factory_anomaly.ml.drift import (
    classify_psi,
    compute_ks,
    compute_psi,
)


def test_ks_identical_distributions_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(1_000)
    assert compute_ks(sample, sample) < 1e-6


def test_ks_disjoint_distributions_is_one() -> None:
    a = np.zeros(500)
    b = np.ones(500)
    assert compute_ks(a, b) == pytest.approx(1.0)


def test_ks_handles_multi_column_and_returns_worst_case() -> None:
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((500, 3))
    current = reference.copy()
    current[:, 1] += 5.0  # shift one column heavily
    result = compute_ks(reference, current)
    assert 0.9 < result <= 1.0


def test_psi_identical_distributions_is_near_zero() -> None:
    rng = np.random.default_rng(1)
    sample = rng.standard_normal(1_000)
    psi = compute_psi(sample, sample)
    assert psi < 0.01


def test_psi_strong_shift_classified_as_strong() -> None:
    rng = np.random.default_rng(1)
    reference = rng.standard_normal(1_000)
    current = rng.standard_normal(1_000) + 5.0
    psi = compute_psi(reference, current)
    assert classify_psi(psi) == "strong"


def test_psi_grows_with_shift_magnitude() -> None:
    rng = np.random.default_rng(2)
    reference = rng.standard_normal(2_000)
    weak = compute_psi(reference, rng.standard_normal(2_000) + 0.2)
    strong = compute_psi(reference, rng.standard_normal(2_000) + 3.0)
    assert weak < strong
    assert classify_psi(strong) == "strong"


def test_psi_handles_multi_column() -> None:
    rng = np.random.default_rng(3)
    reference = rng.standard_normal((1_000, 4))
    current = reference.copy()
    current[:, 2] += 4.0
    psi = compute_psi(reference, current)
    assert psi > 1.0


def test_compute_ks_rejects_mismatched_ndim() -> None:
    with pytest.raises(ValueError, match="same ndim"):
        compute_ks(np.zeros(10), np.zeros((10, 2)))


def test_compute_psi_rejects_mismatched_feature_count() -> None:
    with pytest.raises(ValueError, match="feature counts differ"):
        compute_psi(np.zeros((10, 3)), np.zeros((10, 4)))


def test_compute_psi_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_psi(np.zeros(0), np.zeros(0))


def test_compute_psi_rejects_too_few_bins() -> None:
    with pytest.raises(ValueError, match="bins must be"):
        compute_psi(np.zeros(10), np.zeros(10), bins=1)


def test_classify_psi_thresholds() -> None:
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "moderate"
    assert classify_psi(0.5) == "strong"
