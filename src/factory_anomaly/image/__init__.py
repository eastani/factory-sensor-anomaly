"""Image anomaly module — PatchCore detector.

See [ADR-0006](../../../docs/adr/0006-image-anomaly-modality.md) for the
design rationale (model choice, hybrid-library decision, streaming bridge).

This module is **opt-in**: it requires the `image` dependency group
(`uv sync --group image`). The default sync does not install torch, so
importing this package from a context that lacks torch will raise
``ModuleNotFoundError``. Catch and surface that as a clear "image
service not enabled" error in the calling layer.
"""

from factory_anomaly.image.client import ImageApiClient, ImageApiClientError
from factory_anomaly.image.coreset import greedy_coreset
from factory_anomaly.image.detector import (
    PatchCoreDetector,
    PatchCoreMetadata,
    PatchCoreVersionMismatchError,
)
from factory_anomaly.image.feature_extractor import PatchFeatureExtractor

__all__ = [
    "ImageApiClient",
    "ImageApiClientError",
    "PatchCoreDetector",
    "PatchCoreMetadata",
    "PatchCoreVersionMismatchError",
    "PatchFeatureExtractor",
    "greedy_coreset",
]
