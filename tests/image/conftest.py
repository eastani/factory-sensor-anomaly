"""Fixtures local to the image-module tests.

These tests run in the ``image-quality`` CI job, which installs the
``image`` dependency group. They do **not** share fixtures with the
top-level ``tests/conftest.py`` (testcontainers / Postgres) — the
image module has no database dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from factory_anomaly.image.feature_extractor import FeatureExtractorConfig


@pytest.fixture(scope="session")
def small_feature_config() -> FeatureExtractorConfig:
    """Small spatial size keeps ResNet50 forward + coreset cheap in tests."""
    return FeatureExtractorConfig(layers=("layer2", "layer3"), neighborhood=3, target_spatial=4)


@pytest.fixture
def random_normalised_images() -> torch.Tensor:
    """``(4, 3, 64, 64)`` ImageNet-normalised noise — enough to exercise the
    ResNet50 forward without burning CPU time. Deterministic via fixed seed."""
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((4, 3, 64, 64)).astype(np.float32)
    return torch.from_numpy(raw)
