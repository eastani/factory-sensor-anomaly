"""Tests for the IsolationForest anomaly detector and its persistence layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from factory_anomaly.data import make_example_dataset
from factory_anomaly.ml import AnomalyDetector, SklearnVersionMismatchError
from factory_anomaly.ml.detector import MODEL_NAME, ModelMetadata
from factory_anomaly.ml.features import make_rolling_features


@pytest.fixture
def fitted_detector() -> AnomalyDetector:
    """A detector trained on the canonical example dataset."""
    sig = make_example_dataset(seed=42)
    features = make_rolling_features(sig.values, window=20).to_numpy()
    detector = AnomalyDetector(model_version="test-v1")
    detector.fit(features)
    return detector


def test_unfitted_detector_rejects_score_and_predict() -> None:
    detector = AnomalyDetector(model_version="test-v1")
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.score(np.zeros((10, 5)))
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.predict(np.zeros((10, 5)))
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = detector.metadata


def test_empty_model_version_rejected() -> None:
    with pytest.raises(ValueError, match="model_version is required"):
        AnomalyDetector(model_version="")


def test_fit_records_metadata(fitted_detector: AnomalyDetector) -> None:
    meta = fitted_detector.metadata
    assert meta.model_name == MODEL_NAME
    assert meta.model_version == "test-v1"
    assert meta.training_data_shape[1] == 5  # 5 rolling features
    assert meta.training_data_hash  # non-empty hex
    assert meta.hyperparameters["n_estimators"] == 100


def test_score_returns_per_sample_floats(fitted_detector: AnomalyDetector) -> None:
    features = np.random.default_rng(0).standard_normal((50, 5))
    scores = fitted_detector.score(features)
    assert scores.shape == (50,)
    assert scores.dtype == np.float64


def test_score_flags_extreme_input_higher_than_normal(
    fitted_detector: AnomalyDetector,
) -> None:
    """Inputs far outside the training distribution should score higher."""
    normal_features = make_rolling_features(
        make_example_dataset(seed=0).values, window=20
    ).to_numpy()
    extreme_features = normal_features.copy()
    extreme_features[:5] = 1_000.0

    scores = fitted_detector.score(extreme_features[:20])
    # The first 5 (extreme) rows should score higher than the rest on average.
    assert scores[:5].mean() > scores[5:].mean()


def test_predict_returns_bool_array(fitted_detector: AnomalyDetector) -> None:
    features = np.random.default_rng(0).standard_normal((10, 5))
    preds = fitted_detector.predict(features)
    assert preds.dtype == np.bool_
    assert preds.shape == (10,)


def test_save_writes_artifact_and_sidecar(fitted_detector: AnomalyDetector, tmp_path: Path) -> None:
    target = tmp_path / "detector.joblib"
    written = fitted_detector.save(target)
    sidecar = target.with_suffix(".joblib.meta.json")

    assert written == target
    assert target.exists()
    assert sidecar.exists()

    raw = json.loads(sidecar.read_text())
    assert raw["model_name"] == MODEL_NAME
    assert raw["model_version"] == "test-v1"


def test_load_round_trip_preserves_scores(fitted_detector: AnomalyDetector, tmp_path: Path) -> None:
    target = tmp_path / "detector.joblib"
    fitted_detector.save(target)

    sample = np.random.default_rng(0).standard_normal((20, 5))
    original_scores = fitted_detector.score(sample)

    loaded = AnomalyDetector.load(target)
    np.testing.assert_allclose(loaded.score(sample), original_scores)
    assert loaded.metadata.model_version == "test-v1"


def test_load_rejects_mismatched_sklearn_version_strictly(
    fitted_detector: AnomalyDetector, tmp_path: Path
) -> None:
    target = tmp_path / "detector.joblib"
    fitted_detector.save(target)

    sidecar = target.with_suffix(".joblib.meta.json")
    raw = json.loads(sidecar.read_text())
    raw["sklearn_version"] = "0.0.0-bogus"
    sidecar.write_text(json.dumps(raw))

    with pytest.raises(SklearnVersionMismatchError):
        AnomalyDetector.load(target)

    # Non-strict load must succeed (escape hatch for migrations).
    loaded = AnomalyDetector.load(target, strict=False)
    assert loaded.metadata.sklearn_version == "0.0.0-bogus"


def test_load_errors_when_files_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model artifact"):
        AnomalyDetector.load(tmp_path / "nope.joblib")


def test_metadata_json_round_trip() -> None:
    meta = ModelMetadata(
        model_name="x",
        model_version="v",
        sklearn_version="1.5",
        python_version="3.12",
        trained_at="2026-05-16T00:00:00+00:00",
        training_data_hash="abc",
        training_data_shape=(10, 3),
        hyperparameters={"k": 1},
    )
    restored = ModelMetadata.from_json(meta.to_json())
    assert restored == meta
