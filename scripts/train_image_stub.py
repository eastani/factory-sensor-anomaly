"""Train a small PatchCore memory bank for docker-compose / cloud demos.

Used by ``Dockerfile.image`` so the image-anomaly service has something
meaningful to load on first boot. **Not** intended for real anomaly
detection — replace with a real per-category bank produced by
``scripts/evaluate_image.py``.

Two training modes:

- ``--images-dir <path>`` — fit on real PNG/JPEG files in the directory
  (recursed). This is the path used by Dockerfile.image against the
  bundled demo images at ``docs/assets/demo-images/train/``, so the
  default dashboard demo shows a heatmap that actually localises the
  injected defects.
- No ``--images-dir`` — synthesise N random-noise images and fit on them.
  Kept as a fallback so the script never silently refuses to produce an
  artifact when the demo directory is missing.

The output artifact is the same on-disk format as a real PatchCore bank,
so any downstream code that loads ``image_bank.joblib`` works against it.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from factory_anomaly.image.detector import PatchCoreDetector
from factory_anomaly.image.feature_extractor import FeatureExtractorConfig

_SUPPORTED = frozenset({".png", ".jpg", ".jpeg"})


def _load_images(paths: Iterable[Path], size: int) -> torch.Tensor:
    """Read + resize + normalise a batch of image files into a tensor."""
    arrays = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arrays.append(np.asarray(img, dtype=np.uint8))
    return PatchCoreDetector.stack_images(arrays)


def _collect_image_paths(root: Path) -> list[Path]:
    """Recursively collect supported image files under ``root``."""
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _SUPPORTED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stub PatchCore bank.")
    parser.add_argument("--out", type=Path, required=True, help="output .joblib path")
    parser.add_argument(
        "--version", default="stub-noise", help="model_version recorded in metadata"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="optional directory of real PNG/JPEG images to fit on (recursive)",
    )
    parser.add_argument("--n-images", type=int, default=8, help="noise fallback only")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--target-spatial", type=int, default=8)
    parser.add_argument("--coreset-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.images_dir is not None and args.images_dir.exists():
        paths = _collect_image_paths(args.images_dir)
        if not paths:
            raise SystemExit(f"no images found under {args.images_dir}")
        tensor = _load_images(paths, size=args.image_size)
        source_label = f"images-dir:{args.images_dir} (n={len(paths)})"
    else:
        rng = np.random.default_rng(args.seed)
        raw = rng.standard_normal((args.n_images, 3, args.image_size, args.image_size)).astype(
            np.float32
        )
        tensor = torch.from_numpy(raw)
        source_label = f"noise (n={args.n_images}, seed={args.seed})"

    detector = PatchCoreDetector(
        model_version=args.version,
        coreset_ratio=args.coreset_ratio,
        feature_config=FeatureExtractorConfig(target_spatial=args.target_spatial),
    )
    detector.fit(tensor)
    path = detector.save(args.out)
    print(
        f"wrote {detector.memory_bank.shape[0]}x{detector.memory_bank.shape[1]} "
        f"stub bank to {path} (model_version={args.version}, source={source_label})"
    )


if __name__ == "__main__":
    main()
