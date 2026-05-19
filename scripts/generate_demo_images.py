"""Generate the synthetic demo images used by the Streamlit dashboard.

These images stand in for MVTec AD samples, which we can't commit (CC BY-NC-SA
4.0; see image LICENSE-NOTICE.md). They look broadly like an industrial
inspection target: a regular grid of circular features on a flat plate, with
small jitter so the training set has some variance.

Defective test samples introduce specific anomalies (missing feature,
diagonal scratch) so the dashboard heatmap has something visually
meaningful to localise.

Outputs:

  docs/assets/demo-images/train/clean_{i}.png        (8 training images)
  docs/assets/demo-images/test/clean.png             (1 normal test image)
  docs/assets/demo-images/test/defect_scratch.png    (diagonal scratch)
  docs/assets/demo-images/test/defect_missing.png    (missing feature)

All deterministic via seeded RNG so commits stay reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# 128x128 RGB. Whole demo set is < 100 KB combined thanks to flat-area PNG
# compression — ResNet50 resamples to 224 anyway so resolution doesn't matter.
SIZE = 128
GRID_N = 5  # 5x5 features
BG = (235, 235, 235)
FG = (40, 40, 40)


def _draw_plate(seed: int, *, missing_index: int | None = None) -> Image.Image:
    """Render a clean plate with optional missing feature.

    ``seed`` introduces sub-pixel jitter to feature positions so the training
    set has the natural variance of a manufactured part. No per-pixel noise —
    flat regions compress dramatically better and the jitter alone is enough
    variance for PatchCore's memory bank.
    """
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(img)

    cell = SIZE / (GRID_N + 1)
    radius = cell * 0.30
    idx = 0
    for r in range(GRID_N):
        for c in range(GRID_N):
            if missing_index is not None and idx == missing_index:
                idx += 1
                continue
            cx = cell * (c + 1) + rng.normal(0.0, 0.4)
            cy = cell * (r + 1) + rng.normal(0.0, 0.4)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=FG)
            idx += 1
    return img


def _draw_scratch(base: Image.Image, seed: int) -> Image.Image:
    """Overlay a diagonal scratch line on a clean plate."""
    rng = np.random.default_rng(seed)
    img = base.copy()
    draw = ImageDraw.Draw(img)
    x0 = int(rng.integers(10, 30))
    y0 = int(rng.integers(90, 110))
    x1 = int(rng.integers(90, 115))
    y1 = int(rng.integers(20, 45))
    draw.line((x0, y0, x1, y1), fill=FG, width=2)
    return img


def generate(out_root: Path) -> dict[str, Path]:
    """Write the canonical demo set; return path map for testability."""
    paths: dict[str, Path] = {}
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for i in range(8):
        p = train_dir / f"clean_{i}.png"
        _draw_plate(seed=i).save(p, optimize=True)
        paths[f"train_{i}"] = p

    clean_p = test_dir / "clean.png"
    _draw_plate(seed=100).save(clean_p, optimize=True)
    paths["test_clean"] = clean_p

    scratch_p = test_dir / "defect_scratch.png"
    _draw_scratch(_draw_plate(seed=101), seed=101).save(scratch_p, optimize=True)
    paths["test_scratch"] = scratch_p

    # Missing feature: drop the central (index 12 in a 5x5 row-major grid).
    missing_p = test_dir / "defect_missing.png"
    _draw_plate(seed=102, missing_index=12).save(missing_p, optimize=True)
    paths["test_missing"] = missing_p

    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/demo-images"),
        help="output root (default: docs/assets/demo-images)",
    )
    args = parser.parse_args()
    paths = generate(args.out)
    print(f"Generated {len(paths)} images:")
    for k, p in paths.items():
        print(f"  {k}: {p}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
