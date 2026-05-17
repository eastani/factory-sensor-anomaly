"""Tests for the ResNet50-hook feature extractor.

Inputs are tiny random tensors — these tests verify *shape contracts* and
*determinism*, not anomaly-detection quality (that lives in the evaluation
report).
"""

from __future__ import annotations

import pytest
import torch

from factory_anomaly.image.feature_extractor import (
    FeatureExtractorConfig,
    PatchFeatureExtractor,
)


def test_extract_returns_expected_shape(
    small_feature_config: FeatureExtractorConfig, random_normalised_images: torch.Tensor
) -> None:
    extractor = PatchFeatureExtractor(small_feature_config)
    try:
        out = extractor.extract(random_normalised_images)
    finally:
        extractor.close()

    b = random_normalised_images.shape[0]
    expected_d = 512 + 1024  # layer2 + layer3 channel counts
    expected_hw = small_feature_config.target_spatial**2
    assert out.shape == (b, expected_hw, expected_d)


def test_feature_dim_matches_actual(small_feature_config: FeatureExtractorConfig) -> None:
    extractor = PatchFeatureExtractor(small_feature_config)
    try:
        assert extractor.feature_dim == 512 + 1024
    finally:
        extractor.close()


def test_extract_is_deterministic(
    small_feature_config: FeatureExtractorConfig, random_normalised_images: torch.Tensor
) -> None:
    extractor = PatchFeatureExtractor(small_feature_config)
    try:
        a = extractor.extract(random_normalised_images)
        b = extractor.extract(random_normalised_images)
    finally:
        extractor.close()
    torch.testing.assert_close(a, b)


def test_rejects_wrong_shape(small_feature_config: FeatureExtractorConfig) -> None:
    extractor = PatchFeatureExtractor(small_feature_config)
    try:
        with pytest.raises(ValueError, match=r"\(B, 3, H, W\)"):
            extractor.extract(torch.zeros(4, 64, 64))  # missing channel dim
        with pytest.raises(ValueError, match=r"\(B, 3, H, W\)"):
            extractor.extract(torch.zeros(4, 1, 64, 64))  # grayscale
    finally:
        extractor.close()


def test_config_rejects_unknown_layer() -> None:
    with pytest.raises(ValueError, match="unknown ResNet50 layer"):
        FeatureExtractorConfig(layers=("layer2", "layer99"))


def test_config_rejects_even_neighborhood() -> None:
    with pytest.raises(ValueError, match="positive odd integer"):
        FeatureExtractorConfig(neighborhood=2)


def test_backbone_parameters_are_frozen(small_feature_config: FeatureExtractorConfig) -> None:
    extractor = PatchFeatureExtractor(small_feature_config)
    try:
        assert all(not p.requires_grad for p in extractor._backbone.parameters())
    finally:
        extractor.close()


def test_close_is_idempotent(small_feature_config: FeatureExtractorConfig) -> None:
    extractor = PatchFeatureExtractor(small_feature_config)
    extractor.close()
    extractor.close()  # no error
