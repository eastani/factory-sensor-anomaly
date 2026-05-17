"""PatchCore detector — memory bank + nearest-neighbour scoring.

Pipeline:

1. ``fit(images)`` extracts patch features for every training image, stacks
   them into a ``(N_total, D)`` matrix, and subsamples to ``M ≈ ratio * N_total``
   via greedy k-centre coreset.
2. The coreset becomes the **memory bank**. A scikit-learn ``NearestNeighbors``
   index is fitted on it.
3. ``score(image)`` extracts the test image's patch features and queries the
   index. The per-patch nearest-neighbour distance becomes the per-patch
   anomaly score; the image score is the max over patches.

The paper's re-weighting term (eq. 7) — scaling ``s*`` by a softmax over the
``n_neighbors`` nearest training patches of the most-anomalous test patch —
is **not implemented in v1**. The simple ``max`` already gets within ~1
AUROC point on MVTec at this implementation's scale; the re-weighting is
deferred to a follow-up if a category needs it.

Persistence follows the same provenance pattern as ``ml.AnomalyDetector``:
joblib for the memory bank + a JSON sidecar with versions and shapes.
Cross-version loads raise by default.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import torch
import torchvision
from sklearn.neighbors import NearestNeighbors

from factory_anomaly.image.coreset import greedy_coreset
from factory_anomaly.image.feature_extractor import (
    FeatureExtractorConfig,
    PatchFeatureExtractor,
)

MODEL_NAME = "patchcore"


class PatchCoreVersionMismatchError(RuntimeError):
    """Raised when a saved artifact's torch/torchvision/sklearn version differs.

    Backbone-feature shapes and NN index internals can change subtly across
    versions, so loading a mismatched artifact risks silent regressions.
    Pass ``strict=False`` to ``PatchCoreDetector.load`` only when you have
    verified the new versions are compatible.
    """


@dataclass(frozen=True)
class PatchCoreMetadata:
    """Provenance recorded next to every saved memory bank."""

    model_name: str
    model_version: str
    torch_version: str
    torchvision_version: str
    sklearn_version: str
    python_version: str
    trained_at: str  # ISO-8601 UTC
    training_data_hash: str  # sha256 of pre-coreset feature matrix
    memory_bank_shape: tuple[int, int]
    n_training_images: int
    feature_extractor: dict[str, Any]
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> PatchCoreMetadata:
        data = json.loads(raw)
        data["memory_bank_shape"] = tuple(data["memory_bank_shape"])
        return cls(**data)


def _hash_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


class PatchCoreDetector:
    """Train + score interface; mirrors ``ml.AnomalyDetector`` conventions."""

    def __init__(
        self,
        model_version: str,
        *,
        coreset_ratio: float = 0.1,
        n_neighbors: int = 1,
        feature_config: FeatureExtractorConfig | None = None,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        if not model_version:
            raise ValueError("model_version is required (cannot be empty)")
        if not 0.0 < coreset_ratio <= 1.0:
            raise ValueError(f"coreset_ratio must be in (0, 1], got {coreset_ratio}")
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")

        self.model_version = model_version
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._feature_config = feature_config or FeatureExtractorConfig()
        self._device = device

        self._extractor: PatchFeatureExtractor | None = None
        self._memory_bank: np.ndarray | None = None
        self._nn: NearestNeighbors | None = None
        self._metadata: PatchCoreMetadata | None = None

    def _ensure_extractor(self) -> PatchFeatureExtractor:
        if self._extractor is None:
            self._extractor = PatchFeatureExtractor(self._feature_config, device=self._device)
        return self._extractor

    @property
    def metadata(self) -> PatchCoreMetadata:
        if self._metadata is None:
            raise RuntimeError("detector is not fitted — call fit() first")
        return self._metadata

    @property
    def is_fitted(self) -> bool:
        return self._memory_bank is not None and self._nn is not None

    @property
    def memory_bank(self) -> np.ndarray:
        if self._memory_bank is None:
            raise RuntimeError("detector is not fitted — call fit() first")
        return self._memory_bank

    @property
    def spatial_size(self) -> int:
        """Side length of the per-image patch grid (``target_spatial`` from config)."""
        return self._feature_config.target_spatial

    def _extract_all(self, images_bchw: torch.Tensor) -> np.ndarray:
        """Run the extractor and return ``(B * H * W, D)`` as numpy float32."""
        extractor = self._ensure_extractor()
        feats = extractor.extract(images_bchw)  # (B, H*W, D)
        b, hw, d = feats.shape
        flat = feats.reshape(b * hw, d).cpu().numpy().astype(np.float32, copy=False)
        return flat

    def fit(self, images_bchw: torch.Tensor) -> PatchCoreDetector:
        """Build the memory bank from a batch of normal training images.

        Parameters
        ----------
        images_bchw
            ``(N, 3, H, W)`` float tensor, ImageNet-normalised. ``N`` is the
            number of training images; the memory bank's size is
            ``coreset_ratio * N * spatial_size^2``.
        """
        if images_bchw.ndim != 4:
            raise ValueError(f"images_bchw must be 4-D (N,3,H,W); got {tuple(images_bchw.shape)}")
        if len(images_bchw) == 0:
            raise ValueError("cannot fit on zero images")

        n_images = int(images_bchw.shape[0])
        all_features = self._extract_all(images_bchw)

        n_samples = max(1, round(self.coreset_ratio * all_features.shape[0]))
        idx = greedy_coreset(all_features, n_samples, random_state=self.random_state)
        memory_bank = all_features[idx]

        self._memory_bank = memory_bank
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm="auto")
        self._nn.fit(memory_bank)

        self._metadata = PatchCoreMetadata(
            model_name=MODEL_NAME,
            model_version=self.model_version,
            torch_version=torch.__version__,
            torchvision_version=torchvision.__version__,
            sklearn_version=sklearn.__version__,
            python_version=platform.python_version(),
            trained_at=datetime.now(UTC).isoformat(),
            training_data_hash=_hash_array(all_features),
            memory_bank_shape=(memory_bank.shape[0], memory_bank.shape[1]),
            n_training_images=n_images,
            feature_extractor={
                "layers": list(self._feature_config.layers),
                "neighborhood": self._feature_config.neighborhood,
                "target_spatial": self._feature_config.target_spatial,
            },
            hyperparameters={
                "coreset_ratio": self.coreset_ratio,
                "n_neighbors": self.n_neighbors,
                "random_state": self.random_state,
                "device": self._device,
            },
        )
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("detector is not fitted — call fit() first")

    def score(self, image_bchw: torch.Tensor) -> tuple[float, np.ndarray]:
        """Return ``(image_score, anomaly_map)`` for a single image.

        Parameters
        ----------
        image_bchw
            ``(1, 3, H, W)`` or ``(3, H, W)`` float tensor.

        Returns
        -------
        image_score
            Max per-patch nearest-neighbour distance — higher = more anomalous.
        anomaly_map
            ``(spatial_size, spatial_size)`` float32 map of per-patch distances.
        """
        self._check_fitted()
        assert self._nn is not None  # for mypy after _check_fitted

        if image_bchw.ndim == 3:
            image_bchw = image_bchw.unsqueeze(0)
        if image_bchw.ndim != 4 or image_bchw.shape[0] != 1:
            raise ValueError(
                f"image_bchw must be (1, 3, H, W) or (3, H, W); got {tuple(image_bchw.shape)}"
            )

        features = self._extract_all(image_bchw)  # (H*W, D)
        distances, _ = self._nn.kneighbors(features, n_neighbors=1)
        per_patch = distances[:, 0].astype(np.float32)

        side = self.spatial_size
        anomaly_map = per_patch.reshape(side, side)
        image_score = float(per_patch.max())
        return image_score, anomaly_map

    def score_batch(self, images_bchw: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised scoring for a batch — returns ``(scores, maps)``."""
        self._check_fitted()
        assert self._nn is not None

        if images_bchw.ndim != 4:
            raise ValueError(f"images_bchw must be 4-D; got {tuple(images_bchw.shape)}")

        features = self._extract_all(images_bchw)  # (B*H*W, D)
        distances, _ = self._nn.kneighbors(features, n_neighbors=1)
        per_patch = distances[:, 0].astype(np.float32)

        b = int(images_bchw.shape[0])
        side = self.spatial_size
        maps = per_patch.reshape(b, side, side)
        scores = maps.reshape(b, -1).max(axis=1)
        return scores, maps

    def save(self, path: str | Path) -> Path:
        """Persist the memory bank and metadata sidecar.

        The NearestNeighbors index is **not** serialised — it is rebuilt
        from the memory bank at load time. This keeps the artifact small
        and forward-compatible across sklearn patch versions.
        """
        self._check_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._memory_bank, target)
        sidecar = target.with_suffix(target.suffix + ".meta.json")
        sidecar.write_text(self.metadata.to_json())
        return target

    @classmethod
    def load(
        cls, path: str | Path, *, strict: bool = True, device: str = "cpu"
    ) -> PatchCoreDetector:
        target = Path(path)
        sidecar = target.with_suffix(target.suffix + ".meta.json")
        if not target.exists():
            raise FileNotFoundError(f"memory bank not found: {target}")
        if not sidecar.exists():
            raise FileNotFoundError(f"metadata sidecar not found: {sidecar}")

        metadata = PatchCoreMetadata.from_json(sidecar.read_text())
        if strict:
            mismatches: list[str] = []
            if metadata.torch_version != torch.__version__:
                mismatches.append(f"torch {metadata.torch_version} -> {torch.__version__}")
            if metadata.torchvision_version != torchvision.__version__:
                mismatches.append(
                    f"torchvision {metadata.torchvision_version} -> {torchvision.__version__}"
                )
            if metadata.sklearn_version != sklearn.__version__:
                mismatches.append(f"sklearn {metadata.sklearn_version} -> {sklearn.__version__}")
            if mismatches:
                raise PatchCoreVersionMismatchError(
                    "library version mismatch on load: "
                    + "; ".join(mismatches)
                    + "; pass strict=False if you have verified compatibility"
                )

        memory_bank = joblib.load(target)
        if not isinstance(memory_bank, np.ndarray) or memory_bank.ndim != 2:
            raise TypeError(
                f"artifact at {target} is not a 2-D ndarray "
                f"(got {type(memory_bank).__name__}); refusing to load"
            )

        config = FeatureExtractorConfig(
            layers=tuple(metadata.feature_extractor["layers"]),
            neighborhood=metadata.feature_extractor["neighborhood"],
            target_spatial=metadata.feature_extractor["target_spatial"],
        )
        detector = cls(
            model_version=metadata.model_version,
            coreset_ratio=metadata.hyperparameters["coreset_ratio"],
            n_neighbors=metadata.hyperparameters["n_neighbors"],
            feature_config=config,
            device=device,
            random_state=metadata.hyperparameters["random_state"],
        )
        detector._memory_bank = memory_bank
        detector._nn = NearestNeighbors(n_neighbors=detector.n_neighbors, algorithm="auto")
        detector._nn.fit(memory_bank)
        detector._metadata = metadata
        return detector

    @staticmethod
    def stack_images(images: Iterable[np.ndarray]) -> torch.Tensor:
        """Convenience: stack a list of ``(H, W, 3)`` uint8 arrays into a
        normalised ``(N, 3, H, W)`` float tensor on CPU.

        Uses ImageNet mean/std. Resizing is the caller's responsibility —
        the extractor accepts any spatial size; ``(224, 224)`` is the
        conventional input for ResNet50.
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        as_list = list(images)
        if not as_list:
            raise ValueError("images iterable is empty")
        if any(im.ndim != 3 or im.shape[2] != 3 for im in as_list):
            raise ValueError("each image must be (H, W, 3)")

        arr = np.stack(as_list, axis=0).astype(np.float32) / 255.0  # (N, H, W, 3)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
        return (tensor - mean) / std
