"""Isolation Forest anomaly detector with versioned persistence.

Design notes (see ADR-0002):

- A fitted estimator is **opaque** to operators — they cannot tell which
  sklearn version trained it, which data it saw, or whether it was retrained
  this morning or last quarter. To fix that, every saved artifact has a
  JSON sidecar with full provenance.
- Loading a model trained on a different sklearn version is a **hard error**
  by default. The scikit-learn docs explicitly warn that cross-version
  pickles can deserialise without warning and produce incorrect outputs
  (https://scikit-learn.org/1.3/model_persistence.html).
- Persistence uses ``joblib`` (more efficient than ``pickle`` for fitted
  numpy-heavy estimators).
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

MODEL_NAME = "isolation_forest"


class SklearnVersionMismatchError(RuntimeError):
    """Raised when an artifact was trained on a different sklearn version.

    Loading anyway risks silently incorrect predictions; callers who really
    want to proceed must pass ``strict=False`` to ``AnomalyDetector.load``.
    """


@dataclass(frozen=True)
class ModelMetadata:
    """Provenance recorded alongside every saved model artifact."""

    model_name: str
    model_version: str
    sklearn_version: str
    python_version: str
    trained_at: str  # ISO-8601 UTC
    training_data_hash: str  # sha256 hex
    training_data_shape: tuple[int, int]
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> ModelMetadata:
        data = json.loads(raw)
        # Tuples don't survive JSON — restore from list.
        data["training_data_shape"] = tuple(data["training_data_shape"])
        return cls(**data)


def _hash_array(arr: np.ndarray) -> str:
    """Stable sha256 of a numpy array's bytes (ignores stride, includes dtype/shape)."""
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


class AnomalyDetector:
    """Thin wrapper around scikit-learn's ``IsolationForest``.

    Higher anomaly score = more anomalous (the raw sklearn score is the
    *opposite* convention, so it is negated on the way out). This is the only
    convention exposed to the rest of the codebase.
    """

    def __init__(
        self,
        model_version: str,
        *,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 42,
    ) -> None:
        if not model_version:
            raise ValueError("model_version is required (cannot be empty)")
        self.model_version = model_version
        self._estimator = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._metadata: ModelMetadata | None = None

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise RuntimeError("detector is not fitted — call fit() first")
        return self._metadata

    @property
    def is_fitted(self) -> bool:
        return self._metadata is not None

    def fit(self, features: np.ndarray) -> AnomalyDetector:
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {features.shape}")
        if len(features) == 0:
            raise ValueError("cannot fit on empty feature matrix")

        self._estimator.fit(features)
        self._metadata = ModelMetadata(
            model_name=MODEL_NAME,
            model_version=self.model_version,
            sklearn_version=sklearn.__version__,
            python_version=platform.python_version(),
            trained_at=datetime.now(UTC).isoformat(),
            training_data_hash=_hash_array(features),
            training_data_shape=(features.shape[0], features.shape[1]),
            hyperparameters={
                "n_estimators": self._estimator.n_estimators,
                "contamination": self._estimator.contamination,
                "random_state": self._estimator.random_state,
            },
        )
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("detector is not fitted — call fit() first")

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return anomaly scores — higher = more anomalous."""
        self._check_fitted()
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {features.shape}")
        # sklearn convention: higher = more normal. Negate to flip.
        scores: np.ndarray = -self._estimator.score_samples(features)
        return scores

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return a boolean array — True where ``features`` is judged anomalous."""
        self._check_fitted()
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {features.shape}")
        # sklearn returns -1 (anomaly) / +1 (normal); collapse to bool.
        labels: np.ndarray = self._estimator.predict(features)
        return labels == -1  # type: ignore[no-any-return]

    def save(self, path: str | Path) -> Path:
        """Persist the estimator and its metadata sidecar."""
        self._check_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._estimator, target)
        sidecar = target.with_suffix(target.suffix + ".meta.json")
        sidecar.write_text(self.metadata.to_json())
        return target

    @classmethod
    def load(cls, path: str | Path, *, strict: bool = True) -> AnomalyDetector:
        """Restore a detector from disk, refusing cross-version artifacts by default."""
        target = Path(path)
        sidecar = target.with_suffix(target.suffix + ".meta.json")
        if not target.exists():
            raise FileNotFoundError(f"model artifact not found: {target}")
        if not sidecar.exists():
            raise FileNotFoundError(f"metadata sidecar not found: {sidecar}")

        metadata = ModelMetadata.from_json(sidecar.read_text())
        if strict and metadata.sklearn_version != sklearn.__version__:
            raise SklearnVersionMismatchError(
                f"artifact was trained on sklearn {metadata.sklearn_version}, "
                f"current process has {sklearn.__version__}; "
                "pass strict=False to load anyway (NOT recommended for production)"
            )

        estimator = joblib.load(target)
        if not isinstance(estimator, IsolationForest):
            raise TypeError(
                f"artifact at {target} is {type(estimator).__name__}, not IsolationForest"
            )

        detector = cls(model_version=metadata.model_version)
        detector._estimator = estimator
        detector._metadata = metadata
        return detector
