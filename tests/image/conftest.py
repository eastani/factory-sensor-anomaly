"""Fixtures local to the image-module tests.

These tests run in the ``image-quality`` CI job, which installs the
``image`` dependency group. They do **not** share fixtures with the
top-level ``tests/conftest.py`` (testcontainers / Postgres) — the
image module has no database dependency.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from factory_anomaly.image.detector import PatchCoreDetector
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


@pytest.fixture(scope="session")
def trained_image_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train and persist a tiny PatchCore bank once per session.

    Used by the image API fixture so each test does not pay the ResNet50
    forward pass on startup.
    """
    target = tmp_path_factory.mktemp("image-models") / "bank.joblib"
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((4, 3, 64, 64)).astype(np.float32)
    detector = PatchCoreDetector(
        model_version="test-image-v1",
        coreset_ratio=0.25,
        feature_config=FeatureExtractorConfig(target_spatial=4),
    )
    detector.fit(torch.from_numpy(raw))
    detector.save(target)
    return target


@pytest.fixture
def image_api_client(
    trained_image_model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """FastAPI TestClient wired to a real (tiny) PatchCore bank."""
    monkeypatch.setenv("IMAGE_API_MODEL_PATH", str(trained_image_model_path))
    monkeypatch.setenv("IMAGE_API_INPUT_SIZE", "64")
    monkeypatch.setenv("IMAGE_API_TORCH_NUM_THREADS", "1")

    from factory_anomaly.config import get_image_api_settings

    get_image_api_settings.cache_clear()

    from factory_anomaly.image.api import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client

    get_image_api_settings.cache_clear()
