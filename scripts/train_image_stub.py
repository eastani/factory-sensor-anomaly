"""Train a tiny PatchCore memory bank on synthesised noise.

For docker-compose demos, CI smoke, and the Dockerfile.image build step —
**not** for real anomaly detection. Replace with a bank trained on actual
data (e.g. ``make image-train CATEGORY=bottle`` once Phase 3.3 lands a
``scripts/download_mvtec.py``).

The output artifact is the same on-disk format as a real PatchCore bank,
so any downstream code that loads ``image_bank.joblib`` works against it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from factory_anomaly.image.detector import PatchCoreDetector
from factory_anomaly.image.feature_extractor import FeatureExtractorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stub PatchCore bank.")
    parser.add_argument("--out", type=Path, required=True, help="output .joblib path")
    parser.add_argument(
        "--version", default="stub-noise", help="model_version recorded in metadata"
    )
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--target-spatial", type=int, default=4)
    parser.add_argument("--coreset-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = rng.standard_normal((args.n_images, 3, args.image_size, args.image_size)).astype(
        np.float32
    )
    detector = PatchCoreDetector(
        model_version=args.version,
        coreset_ratio=args.coreset_ratio,
        feature_config=FeatureExtractorConfig(target_spatial=args.target_spatial),
    )
    detector.fit(torch.from_numpy(raw))
    path = detector.save(args.out)
    print(
        f"wrote {detector.memory_bank.shape[0]}x{detector.memory_bank.shape[1]} "
        f"stub bank to {path} (model_version={args.version})"
    )


if __name__ == "__main__":
    main()
