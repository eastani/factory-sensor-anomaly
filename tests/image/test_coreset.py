"""Tests for greedy k-centre coreset subsampling."""

from __future__ import annotations

import numpy as np
import pytest

from factory_anomaly.image.coreset import greedy_coreset


def test_returns_requested_count() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((200, 16)).astype(np.float32)
    indices = greedy_coreset(features, n_samples=20)
    assert indices.shape == (20,)
    assert indices.dtype == np.int64


def test_indices_are_unique() -> None:
    rng = np.random.default_rng(1)
    features = rng.standard_normal((100, 8)).astype(np.float32)
    indices = greedy_coreset(features, n_samples=30)
    assert len(set(indices.tolist())) == 30


def test_indices_in_range() -> None:
    rng = np.random.default_rng(2)
    features = rng.standard_normal((50, 4)).astype(np.float32)
    indices = greedy_coreset(features, n_samples=10)
    assert indices.min() >= 0
    assert indices.max() < 50


def test_returns_all_when_oversubsampled() -> None:
    """If n_samples >= N, return all indices in original order — no error."""
    features = np.eye(5).astype(np.float32)
    indices = greedy_coreset(features, n_samples=5)
    np.testing.assert_array_equal(indices, np.arange(5))

    indices_over = greedy_coreset(features, n_samples=99)
    np.testing.assert_array_equal(indices_over, np.arange(5))


def test_deterministic_under_same_seed() -> None:
    rng = np.random.default_rng(3)
    features = rng.standard_normal((80, 12)).astype(np.float32)
    a = greedy_coreset(features, n_samples=15, random_state=7)
    b = greedy_coreset(features, n_samples=15, random_state=7)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_diverge() -> None:
    rng = np.random.default_rng(4)
    features = rng.standard_normal((80, 12)).astype(np.float32)
    a = greedy_coreset(features, n_samples=15, random_state=1)
    b = greedy_coreset(features, n_samples=15, random_state=2)
    # Greedy picks differ when the random seed picks a different starting point.
    assert not np.array_equal(a, b)


def test_covers_well_separated_clusters() -> None:
    """On 3 well-separated clusters, requesting 3 samples should hit all 3."""
    rng = np.random.default_rng(5)
    centres = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]], dtype=np.float32)
    cluster_assignments = rng.integers(0, 3, size=300)
    jitter = rng.standard_normal((300, 2)).astype(np.float32) * 0.5
    features = centres[cluster_assignments] + jitter

    selected = greedy_coreset(features, n_samples=3, random_state=0)
    chosen_clusters = {int(cluster_assignments[i]) for i in selected.tolist()}
    assert chosen_clusters == {0, 1, 2}


def test_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        greedy_coreset(np.zeros(10, dtype=np.float32), n_samples=3)


def test_rejects_zero_n_samples() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        greedy_coreset(np.zeros((10, 4), dtype=np.float32), n_samples=0)
