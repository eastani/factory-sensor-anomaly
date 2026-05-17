"""Tests for the image-ingester worker."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from factory_anomaly.image.workers import (
    ImageIngesterConfig,
    list_source_images,
    run_one_tick,
    select_image_bytes,
    synthesise_noise_png,
)


class _FakeClient:
    """In-memory stand-in for ImageApiClient — captures calls."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[bytes, str]] = []
        self.fail = fail

    def predict(self, image_bytes: bytes, *, filename: str = "image.png") -> dict[str, Any]:
        self.calls.append((image_bytes, filename))
        if self.fail:
            from factory_anomaly.image.client import ImageApiClientError

            raise ImageApiClientError("simulated failure", status_code=500)
        return {"score": 1.5, "model_version": "v", "elapsed_ms": 12.3}


def test_synthesise_noise_png_decodes_to_target_size() -> None:
    raw = synthesise_noise_png(seed=42, size=32)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (32, 32)
    assert img.format == "PNG"


def test_synthesise_noise_is_deterministic() -> None:
    a = synthesise_noise_png(seed=7, size=16)
    b = synthesise_noise_png(seed=7, size=16)
    assert a == b


def test_list_source_images_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "z.png").write_bytes(b"")
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "ignored.txt").write_bytes(b"")
    (tmp_path / "b.jpeg").write_bytes(b"")

    result = list_source_images(tmp_path)
    assert [p.name for p in result] == ["a.jpg", "b.jpeg", "z.png"]


def test_list_source_images_returns_empty_when_missing(tmp_path: Path) -> None:
    assert list_source_images(tmp_path / "nope") == []
    assert list_source_images(None) == []


def test_select_image_bytes_uses_directory_when_available(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    target = tmp_path / "real.png"
    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(target)

    cfg = ImageIngesterConfig(source_dir=tmp_path)
    raw, label = select_image_bytes(cfg, tick=0, seed=0)
    assert raw == target.read_bytes()
    assert label == "file:real.png"


def test_select_image_bytes_cycles_through_files(tmp_path: Path) -> None:
    for i in range(3):
        Image.new("RGB", (8, 8), color=(i, i, i)).save(tmp_path / f"img-{i}.png")
    cfg = ImageIngesterConfig(source_dir=tmp_path)
    files = list_source_images(tmp_path)

    labels = [select_image_bytes(cfg, tick=t, seed=0, cached_files=files)[1] for t in range(6)]
    # Three files, six ticks → each appears twice.
    assert sorted(labels) == sorted([f"file:img-{i}.png" for i in range(3)] * 2)


def test_select_image_bytes_falls_back_to_noise(tmp_path: Path) -> None:
    cfg = ImageIngesterConfig(source_dir=tmp_path / "missing", fallback_size=16)
    raw, label = select_image_bytes(cfg, tick=0, seed=99)
    assert label == "synthesised:noise"
    img = Image.open(io.BytesIO(raw))
    assert img.size == (16, 16)


def test_run_one_tick_posts_and_returns_true() -> None:
    cfg = ImageIngesterConfig(source_dir=None, fallback_size=16)
    client = _FakeClient()
    assert run_one_tick(client, cfg, tick=1, seed=0) is True  # type: ignore[arg-type]
    assert len(client.calls) == 1
    _, filename = client.calls[0]
    assert filename == "tick-1.png"


def test_run_one_tick_returns_false_on_api_error() -> None:
    cfg = ImageIngesterConfig(source_dir=None, fallback_size=16)
    client = _FakeClient(fail=True)
    assert run_one_tick(client, cfg, tick=2, seed=0) is False  # type: ignore[arg-type]


def test_config_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IMAGE_INGESTER_SOURCE", str(tmp_path))
    monkeypatch.setenv("IMAGE_INGESTER_FALLBACK_SIZE", "48")
    monkeypatch.setenv("IMAGE_INGESTER_INTERVAL_SECONDS", "7")
    cfg = ImageIngesterConfig.from_env()
    assert cfg.source_dir == tmp_path
    assert cfg.fallback_size == 48
    assert cfg.interval_seconds == 7.0


def test_config_from_env_with_blank_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_INGESTER_SOURCE", "")
    cfg = ImageIngesterConfig.from_env()
    assert cfg.source_dir is None
