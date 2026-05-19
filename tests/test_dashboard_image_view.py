"""Unit tests for the dashboard's image-anomaly helpers.

These functions are deliberately torch-free so they live in the default
test lane. The dashboard container runs without PyTorch and talks to the
image API over HTTP; this module must keep that invariant.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from factory_anomaly.dashboard.image_view import (
    compose_heatmap_figure,
    downsample_for_upload,
    is_stub_model,
    list_demo_images,
)


def _png_bytes(*, size: tuple[int, int], color: tuple[int, int, int] = (200, 200, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_list_demo_images_handles_missing_root(tmp_path: Path) -> None:
    assert list_demo_images(tmp_path / "nope") == []


def test_list_demo_images_returns_label_sorted(tmp_path: Path) -> None:
    for name in ("clean.png", "defect_scratch.png", "defect_missing.png", "other.png"):
        Image.new("RGB", (64, 64)).save(tmp_path / name)
    demos = list_demo_images(tmp_path)
    names = [d.path.name for d in demos]
    assert names == sorted(names)
    labels = {d.path.name: d.label for d in demos}
    assert "no defect" in labels["clean.png"].lower()
    assert "scratch" in labels["defect_scratch.png"].lower()
    # Unknown filename falls back to the raw filename as label.
    assert labels["other.png"] == "other.png"


def test_downsample_for_upload_passthrough_for_small_images() -> None:
    raw = _png_bytes(size=(200, 200))
    out = downsample_for_upload(raw, max_size=512)
    # Output is normalised to PNG re-encode; size still matches.
    img = Image.open(io.BytesIO(out))
    assert img.size == (200, 200)


def test_downsample_for_upload_resizes_large_images() -> None:
    raw = _png_bytes(size=(2048, 1024))
    out = downsample_for_upload(raw, max_size=512)
    img = Image.open(io.BytesIO(out))
    # Aspect ratio preserved; longest side equals max_size.
    assert max(img.size) == 512
    assert img.size[0] / img.size[1] == pytest.approx(2.0, rel=0.05)


def test_downsample_for_upload_rejects_garbage() -> None:
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        downsample_for_upload(b"not-an-image")


def _suptitle_text(fig) -> str:  # type: ignore[no-untyped-def]
    """Pull the figure's suptitle text via the public Text-artist surface."""
    return " ".join(t.get_text() for t in fig.texts)


def test_compose_heatmap_figure_returns_three_axes() -> None:
    raw = _png_bytes(size=(128, 128))
    anomaly_map = np.random.default_rng(0).random((14, 14)).astype(np.float32)
    fig = compose_heatmap_figure(raw, anomaly_map, score=0.5, model_version="test-v1")
    assert len(fig.axes) == 3
    title = _suptitle_text(fig)
    assert "0.500" in title
    assert "test-v1" in title


def test_compose_heatmap_figure_omits_model_when_empty() -> None:
    raw = _png_bytes(size=(64, 64))
    anomaly_map = np.zeros((8, 8), dtype=np.float32)
    fig = compose_heatmap_figure(raw, anomaly_map, score=0.0)
    assert "model:" not in _suptitle_text(fig)


def test_compose_heatmap_figure_rejects_non_2d_map() -> None:
    raw = _png_bytes(size=(64, 64))
    bad = np.zeros((2, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        compose_heatmap_figure(raw, bad, score=0.0)


def test_is_stub_model() -> None:
    assert is_stub_model("stub-demo-20260517")
    assert is_stub_model("stub-noise")
    assert is_stub_model("stub-baked-20260516")
    assert not is_stub_model("mvtec-capsule-v1")
    assert not is_stub_model("")
    assert not is_stub_model(None)
