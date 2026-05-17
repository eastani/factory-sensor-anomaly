"""Smoke tests for the evaluation pipeline.

We never download MVTec in CI (license + size). Instead we build a tiny
fake MVTec-layout directory from synthesised images and verify the pipeline
end-to-end: train → save → evaluate → JSON. Numeric values are not asserted
beyond shape and range; this is a structural smoke test.
"""

from __future__ import annotations

import json

# Import via the scripts directory so we exercise the same code an operator runs.
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import evaluate_image  # noqa: E402


def _save_random_png(path: Path, *, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _save_random_mask(path: Path, *, size: int, seed: int) -> None:
    """Single-channel mask with a small defective region."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size), dtype=np.uint8)
    # Random rectangle as the "defect".
    x0, y0 = int(rng.integers(0, size // 2)), int(rng.integers(0, size // 2))
    arr[y0 : y0 + size // 4, x0 : x0 + size // 4] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def fake_mvtec(tmp_path: Path) -> Path:
    """Build a minimal MVTec layout with one category."""
    root = tmp_path / "mvtec"
    cat = root / "widget"

    for i in range(4):
        _save_random_png(cat / "train" / "good" / f"{i:03d}.png", size=64, seed=i)
    for i in range(2):
        _save_random_png(cat / "test" / "good" / f"{i:03d}.png", size=64, seed=100 + i)
    for i in range(3):
        _save_random_png(cat / "test" / "defect_x" / f"{i:03d}.png", size=64, seed=200 + i)
        _save_random_mask(
            cat / "ground_truth" / "defect_x" / f"{i:03d}_mask.png", size=64, seed=300 + i
        )
    return root


def test_load_train_images_returns_normalised_tensor(fake_mvtec: Path) -> None:
    tensor = evaluate_image.load_train_images(fake_mvtec / "widget", size=64)
    assert tensor.shape == (4, 3, 64, 64)
    assert tensor.dtype == torch.float32


def test_list_test_samples_finds_all_with_labels(fake_mvtec: Path) -> None:
    samples = evaluate_image.list_test_samples(fake_mvtec / "widget")
    assert len(samples) == 5  # 2 good + 3 defective
    labels = [s.label for s in samples]
    assert labels.count(0) == 2
    assert labels.count(1) == 3
    defective = [s for s in samples if s.label == 1]
    assert all(s.mask_path is not None and s.mask_path.exists() for s in defective)


def test_run_full_evaluation_produces_results(fake_mvtec: Path, tmp_path: Path) -> None:
    results = evaluate_image.run_full_evaluation(
        ["widget"],
        data_root=fake_mvtec,
        models_dir=tmp_path / "models",
        size=64,
        target_spatial=4,
        coreset_ratio=0.5,
        cross_category=False,
    )
    assert len(results) == 1
    result = results[0]
    assert result.experiment == "in_domain"
    assert result.train_category == "widget"
    assert result.test_category == "widget"
    assert result.n_train_images == 4
    assert result.n_test_images == 5
    assert result.memory_bank_size > 0
    assert 0.0 <= result.image_auroc <= 1.0
    # Pixel AUROC is optional but should be computed for this fixture (we
    # provided masks).
    assert result.pixel_auroc is not None
    assert 0.0 <= result.pixel_auroc <= 1.0
    assert result.latency_ms["mean"] > 0
    assert result.latency_ms["p95"] >= result.latency_ms["p50"]


def test_cross_category_runs_when_two_categories(fake_mvtec: Path, tmp_path: Path) -> None:
    # Add a second category by symlinking widget → widget2 (cheap fixture).
    second = fake_mvtec / "widget2"
    (second / "train" / "good").mkdir(parents=True)
    (second / "test" / "good").mkdir(parents=True)
    for f in (fake_mvtec / "widget" / "train" / "good").glob("*.png"):
        (second / "train" / "good" / f.name).write_bytes(f.read_bytes())
    for f in (fake_mvtec / "widget" / "test" / "good").glob("*.png"):
        (second / "test" / "good" / f.name).write_bytes(f.read_bytes())
    # Defective copies too.
    src_def = fake_mvtec / "widget" / "test" / "defect_x"
    dst_def = second / "test" / "defect_x"
    dst_def.mkdir(parents=True)
    for f in src_def.glob("*.png"):
        (dst_def / f.name).write_bytes(f.read_bytes())

    results = evaluate_image.run_full_evaluation(
        ["widget", "widget2"],
        data_root=fake_mvtec,
        models_dir=tmp_path / "models",
        size=64,
        target_spatial=4,
        coreset_ratio=0.5,
        cross_category=True,
    )
    # 2 in_domain + 2 cross_category
    assert len(results) == 4
    experiments = sorted({r.experiment for r in results})
    assert experiments == ["cross_category", "in_domain"]


def test_write_results_json_roundtrip(tmp_path: Path) -> None:
    fake_result = evaluate_image.EvalResult(
        experiment="in_domain",
        train_category="widget",
        test_category="widget",
        n_train_images=10,
        n_test_images=5,
        n_defective=3,
        memory_bank_size=200,
        image_auroc=0.92,
        pixel_auroc=0.88,
        latency_ms={"mean": 150.0, "p50": 140.0, "p95": 200.0, "max": 250.0, "n_samples": 5.0},
    )
    target = tmp_path / "report.json"
    evaluate_image.write_results_json([fake_result], target, meta={"k": "v"})
    loaded = json.loads(target.read_text())
    assert loaded["meta"]["k"] == "v"
    assert len(loaded["experiments"]) == 1
    assert loaded["experiments"][0]["image_auroc"] == 0.92
