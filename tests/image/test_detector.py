"""Tests for the PatchCore detector — fit, score, persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from factory_anomaly.image.detector import (
    PatchCoreDetector,
    PatchCoreMetadata,
    PatchCoreVersionMismatchError,
)
from factory_anomaly.image.feature_extractor import FeatureExtractorConfig


@pytest.fixture(scope="module")
def trained_detector() -> PatchCoreDetector:
    """A detector fitted once for the module's read-only tests."""
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((6, 3, 64, 64)).astype(np.float32)
    images = torch.from_numpy(raw)

    detector = PatchCoreDetector(
        model_version="test-v1",
        coreset_ratio=0.25,
        n_neighbors=1,
        feature_config=FeatureExtractorConfig(target_spatial=4),
    )
    detector.fit(images)
    return detector


def test_fit_populates_memory_bank(trained_detector: PatchCoreDetector) -> None:
    assert trained_detector.is_fitted
    bank = trained_detector.memory_bank
    assert bank.ndim == 2
    # 6 images * 4*4 patches * 0.25 ratio = 24 expected
    assert bank.shape[0] == 24
    assert bank.shape[1] == 512 + 1024
    assert bank.dtype == np.float32


def test_metadata_records_provenance(trained_detector: PatchCoreDetector) -> None:
    meta = trained_detector.metadata
    assert meta.model_name == "patchcore"
    assert meta.model_version == "test-v1"
    assert meta.torch_version
    assert meta.torchvision_version
    assert meta.sklearn_version
    assert meta.n_training_images == 6
    assert meta.feature_extractor["target_spatial"] == 4
    assert meta.hyperparameters["coreset_ratio"] == 0.25


def test_score_single_image_returns_expected_shapes(trained_detector: PatchCoreDetector) -> None:
    rng = np.random.default_rng(1)
    test_image = torch.from_numpy(rng.standard_normal((1, 3, 64, 64)).astype(np.float32))
    score, anomaly_map = trained_detector.score(test_image)

    assert isinstance(score, float)
    assert score >= 0.0
    assert anomaly_map.shape == (4, 4)
    assert anomaly_map.dtype == np.float32
    assert float(anomaly_map.max()) == pytest.approx(score)


def test_score_accepts_unbatched_image(trained_detector: PatchCoreDetector) -> None:
    rng = np.random.default_rng(2)
    test_image = torch.from_numpy(rng.standard_normal((3, 64, 64)).astype(np.float32))
    score, _ = trained_detector.score(test_image)
    assert isinstance(score, float)


def test_score_batch_shapes(trained_detector: PatchCoreDetector) -> None:
    rng = np.random.default_rng(3)
    batch = torch.from_numpy(rng.standard_normal((3, 3, 64, 64)).astype(np.float32))
    scores, maps = trained_detector.score_batch(batch)
    assert scores.shape == (3,)
    assert maps.shape == (3, 4, 4)
    # Per-image score is the max of its map.
    np.testing.assert_allclose(scores, maps.reshape(3, -1).max(axis=1), rtol=1e-6)


def test_save_and_load_roundtrip(trained_detector: PatchCoreDetector, tmp_path: Path) -> None:
    target = tmp_path / "memory_bank.joblib"
    trained_detector.save(target)
    assert target.exists()
    assert (target.parent / "memory_bank.joblib.meta.json").exists()

    loaded = PatchCoreDetector.load(target)
    assert loaded.is_fitted
    assert loaded.memory_bank.shape == trained_detector.memory_bank.shape
    np.testing.assert_array_equal(loaded.memory_bank, trained_detector.memory_bank)
    assert loaded.metadata.model_version == "test-v1"


def test_load_scores_consistently_with_fitted(
    trained_detector: PatchCoreDetector, tmp_path: Path
) -> None:
    target = tmp_path / "bank.joblib"
    trained_detector.save(target)
    loaded = PatchCoreDetector.load(target)

    rng = np.random.default_rng(4)
    test_image = torch.from_numpy(rng.standard_normal((1, 3, 64, 64)).astype(np.float32))
    score_a, map_a = trained_detector.score(test_image)
    score_b, map_b = loaded.score(test_image)
    assert score_a == pytest.approx(score_b)
    np.testing.assert_allclose(map_a, map_b, rtol=1e-6)


def test_load_strict_rejects_version_mismatch(
    trained_detector: PatchCoreDetector, tmp_path: Path
) -> None:
    target = tmp_path / "bank.joblib"
    trained_detector.save(target)
    sidecar = target.with_suffix(target.suffix + ".meta.json")

    # Tamper with the recorded torch version so strict-load must reject.
    meta = json.loads(sidecar.read_text())
    meta["torch_version"] = "0.0.0-fake"
    sidecar.write_text(json.dumps(meta))

    with pytest.raises(PatchCoreVersionMismatchError, match="torch"):
        PatchCoreDetector.load(target)

    # strict=False bypasses the gate.
    loaded = PatchCoreDetector.load(target, strict=False)
    assert loaded.is_fitted


def test_metadata_json_roundtrip(trained_detector: PatchCoreDetector) -> None:
    raw = trained_detector.metadata.to_json()
    restored = PatchCoreMetadata.from_json(raw)
    assert restored == trained_detector.metadata


def test_rejects_empty_fit() -> None:
    detector = PatchCoreDetector(model_version="empty")
    with pytest.raises(ValueError, match="cannot fit on zero images"):
        detector.fit(torch.zeros(0, 3, 64, 64))


def test_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError, match="model_version is required"):
        PatchCoreDetector(model_version="")
    with pytest.raises(ValueError, match="coreset_ratio must be in"):
        PatchCoreDetector(model_version="v", coreset_ratio=0.0)
    with pytest.raises(ValueError, match="coreset_ratio must be in"):
        PatchCoreDetector(model_version="v", coreset_ratio=1.5)
    with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
        PatchCoreDetector(model_version="v", n_neighbors=0)


def test_score_before_fit_raises() -> None:
    detector = PatchCoreDetector(model_version="unfit")
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.score(torch.zeros(1, 3, 64, 64))


def test_stack_images_normalises() -> None:
    """uint8 HWC inputs become normalised float32 NCHW tensors."""
    images = [np.full((8, 8, 3), 128, dtype=np.uint8) for _ in range(2)]
    tensor = PatchCoreDetector.stack_images(images)
    assert tensor.shape == (2, 3, 8, 8)
    assert tensor.dtype == torch.float32
    # 128/255 = ~0.502; ImageNet mean ~0.456 -> result is near zero per channel.
    assert tensor.abs().mean().item() < 1.0


def test_stack_images_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        PatchCoreDetector.stack_images([np.zeros((8, 8), dtype=np.uint8)])
    with pytest.raises(ValueError, match="empty"):
        PatchCoreDetector.stack_images([])
