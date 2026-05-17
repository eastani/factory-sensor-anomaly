"""Image-ingester sidecar (Phase 3.2; see ADR-0006).

Mirrors the sensor ``ingester`` pattern from ADR-0004: a small process that
emits one inference request on a fixed cadence so the dashboard always has
something fresh to render. Designed to be restartable — one missed tick is
fine.

Image source resolution order:

1. ``IMAGE_INGESTER_SOURCE`` env points to a directory of PNG/JPG files →
   the worker cycles through them deterministically.
2. Otherwise → falls back to synthesised noise PNGs. This keeps
   ``docker-compose up`` working on a fresh clone with no dataset present
   (and on CI, which has no MVTec download). The dashboard preview will
   look like noise, which is exactly what it should look like when no real
   conveyor camera is wired in.
"""

from __future__ import annotations

import io
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from factory_anomaly.image.client import ImageApiClient, ImageApiClientError
from factory_anomaly.logging_config import configure_logging, get_logger

log = get_logger(__name__)

_SUPPORTED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class ImageIngesterConfig:
    source_dir: Path | None = None
    fallback_size: int = 64  # size of synthesised noise images
    interval_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> ImageIngesterConfig:
        raw_dir = os.environ.get("IMAGE_INGESTER_SOURCE", "").strip()
        source_dir = Path(raw_dir) if raw_dir else None
        return cls(
            source_dir=source_dir,
            fallback_size=int(os.environ.get("IMAGE_INGESTER_FALLBACK_SIZE", "64")),
            interval_seconds=float(os.environ.get("IMAGE_INGESTER_INTERVAL_SECONDS", "15")),
        )


def list_source_images(source_dir: Path | None) -> Sequence[Path]:
    """Return a sorted list of supported image files in ``source_dir``.

    Returns an empty sequence if the directory is missing or empty so the
    caller can fall back to noise without raising.
    """
    if source_dir is None or not source_dir.exists():
        return []
    return sorted(p for p in source_dir.iterdir() if p.suffix.lower() in _SUPPORTED_SUFFIXES)


def synthesise_noise_png(seed: int, size: int) -> bytes:
    """Generate a deterministic-by-seed uint8 noise image, encoded as PNG."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def select_image_bytes(
    cfg: ImageIngesterConfig,
    *,
    tick: int,
    seed: int,
    cached_files: Sequence[Path] | None = None,
) -> tuple[bytes, str]:
    """Return ``(image_bytes, source_label)`` for the given tick.

    ``cached_files`` lets the caller scan the source directory once at
    startup; if ``None``, the directory is scanned on every call.
    """
    files = cached_files if cached_files is not None else list_source_images(cfg.source_dir)
    if files:
        path = files[tick % len(files)]
        return path.read_bytes(), f"file:{path.name}"
    return synthesise_noise_png(seed, cfg.fallback_size), "synthesised:noise"


def run_one_tick(
    client: ImageApiClient,
    cfg: ImageIngesterConfig,
    *,
    tick: int,
    seed: int,
    cached_files: Sequence[Path] | None = None,
) -> bool:
    """Send one image to the image API. Returns True on success."""
    image_bytes, source = select_image_bytes(cfg, tick=tick, seed=seed, cached_files=cached_files)
    try:
        resp = client.predict(image_bytes, filename=f"tick-{tick}.png")
        log.info(
            "image_ingester.posted",
            tick=tick,
            source=source,
            score=resp["score"],
            elapsed_ms=resp["elapsed_ms"],
        )
        return True
    except ImageApiClientError as exc:
        log.warning("image_ingester.api_error", tick=tick, source=source, error=str(exc))
        return False


def main() -> None:  # pragma: no cover — exercised end-to-end via docker compose
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    cfg = ImageIngesterConfig.from_env()
    api_base_url = os.environ.get("IMAGE_API_BASE_URL", "http://image-api:8001")

    cached_files = list_source_images(cfg.source_dir)
    log.info(
        "image_ingester.start",
        api_base_url=api_base_url,
        source_dir=str(cfg.source_dir) if cfg.source_dir else None,
        source_files_count=len(cached_files),
        interval=cfg.interval_seconds,
        fallback_size=cfg.fallback_size,
    )

    tick = 0
    with ImageApiClient(api_base_url, timeout=30.0) as client:
        while True:
            tick += 1
            seed = int(datetime.now(UTC).timestamp() * 1_000) % (2**31 - 1)
            run_one_tick(client, cfg, tick=tick, seed=seed, cached_files=cached_files)
            time.sleep(cfg.interval_seconds)


if __name__ == "__main__":  # pragma: no cover
    main()
