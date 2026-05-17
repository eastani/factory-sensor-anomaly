"""Train + evaluate PatchCore on MVTec AD categories.

Produces:

- One memory-bank artifact per training category (``models/image_<cat>.joblib``)
- A JSON report aggregating image-level AUROC, pixel-level AUROC, and CPU
  inference latency stats across in-domain + cross-category experiments.

The output JSON is the source of truth; the human-readable
``docs/evaluation/image-baseline.md`` is a hand-curated narrative built
around the same numbers.

See ADR-0006 for the model choice and honest-finding plan. The two
findings this script is designed to surface:

1. **Cross-category collapse** — a bank trained on category A and
   evaluated on category B should perform near chance, demonstrating
   the per-category-bank constraint quantitatively.
2. **CPU latency asymmetry** — image inference is 5-20x slower than the
   sensor side; the "streaming" framing only holds at low frame rates.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 — PyTorch-standard alias
from PIL import Image
from sklearn.metrics import roc_auc_score

from factory_anomaly.image.detector import PatchCoreDetector
from factory_anomaly.image.feature_extractor import FeatureExtractorConfig

# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------


def _load_image(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _load_mask(path: Path, size: int) -> np.ndarray:
    """Load a single-channel ground-truth mask, resize, binarise to {0,1}."""
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def load_train_images(category_root: Path, size: int) -> torch.Tensor:
    """Stack all ``train/good`` images of one category into a normalised tensor."""
    train_dir = category_root / "train" / "good"
    paths = sorted(train_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no training images in {train_dir}")
    arrays = [_load_image(p, size) for p in paths]
    return PatchCoreDetector.stack_images(arrays)


@dataclass(frozen=True)
class TestSample:
    image_path: Path
    label: int  # 0 = good, 1 = defective
    defect_type: str  # "good" or e.g. "broken_large"
    mask_path: Path | None  # ground-truth mask (defects only)


def list_test_samples(category_root: Path) -> list[TestSample]:
    """Enumerate the entire test set for one category with labels and masks."""
    test_dir = category_root / "test"
    mask_dir = category_root / "ground_truth"
    samples: list[TestSample] = []
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        is_defective = defect_type != "good"
        for image_path in sorted(defect_dir.glob("*.png")):
            mask_path: Path | None = None
            if is_defective:
                candidate = mask_dir / defect_type / f"{image_path.stem}_mask.png"
                if candidate.exists():
                    mask_path = candidate
            samples.append(
                TestSample(
                    image_path=image_path,
                    label=int(is_defective),
                    defect_type=defect_type,
                    mask_path=mask_path,
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------


def train_detector(
    category_root: Path,
    *,
    size: int,
    target_spatial: int,
    coreset_ratio: float,
    model_version: str,
) -> PatchCoreDetector:
    images = load_train_images(category_root, size)
    detector = PatchCoreDetector(
        model_version=model_version,
        coreset_ratio=coreset_ratio,
        feature_config=FeatureExtractorConfig(target_spatial=target_spatial),
    )
    detector.fit(images)
    return detector


@dataclass(frozen=True)
class EvalResult:
    experiment: str
    train_category: str
    test_category: str
    n_train_images: int
    n_test_images: int
    n_defective: int
    memory_bank_size: int
    image_auroc: float
    pixel_auroc: float | None  # None if no defect masks were available
    latency_ms: dict[str, float]


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def evaluate(
    detector: PatchCoreDetector,
    test_category_root: Path,
    *,
    train_category: str,
    test_category: str,
    n_train_images: int,
    size: int,
    experiment: str,
) -> EvalResult:
    samples = list_test_samples(test_category_root)
    if not samples:
        raise FileNotFoundError(f"no test samples under {test_category_root / 'test'}")

    scores: list[float] = []
    labels: list[int] = []
    latencies_ms: list[float] = []

    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    saw_any_mask = False

    for sample in samples:
        arr = _load_image(sample.image_path, size)
        tensor = PatchCoreDetector.stack_images([arr])

        start = time.perf_counter()
        image_score, anomaly_map = detector.score(tensor)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        scores.append(image_score)
        labels.append(sample.label)

        # Pixel AUROC contribution. For defective samples we use the actual
        # ground-truth mask; for good samples (mask_path is None *and*
        # label==0) we use a zero mask so the metric reflects both true
        # negatives and true positives, matching the paper's convention.
        # Skip entirely if no mask was ever seen — some categories have no
        # ground_truth/ directory in mirrors.
        if sample.mask_path is not None or sample.label == 0:
            heatmap = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0)
            heatmap_up = (
                F.interpolate(heatmap, size=(size, size), mode="bilinear", align_corners=False)
                .squeeze()
                .numpy()
            )
            if sample.mask_path is not None:
                mask = _load_mask(sample.mask_path, size)
                saw_any_mask = True
            else:
                mask = np.zeros((size, size), dtype=np.uint8)
            pixel_scores.append(heatmap_up.reshape(-1))
            pixel_labels.append(mask.reshape(-1))

    image_auroc = float(roc_auc_score(labels, scores))

    pixel_auroc: float | None = None
    if saw_any_mask and pixel_scores:
        pixel_scores_arr = np.concatenate(pixel_scores)
        pixel_labels_arr = np.concatenate(pixel_labels)
        if 0 < pixel_labels_arr.sum() < len(pixel_labels_arr):
            pixel_auroc = float(roc_auc_score(pixel_labels_arr, pixel_scores_arr))

    return EvalResult(
        experiment=experiment,
        train_category=train_category,
        test_category=test_category,
        n_train_images=n_train_images,
        n_test_images=len(samples),
        n_defective=sum(s.label for s in samples),
        memory_bank_size=int(detector.memory_bank.shape[0]),
        image_auroc=image_auroc,
        pixel_auroc=pixel_auroc,
        latency_ms={
            "mean": float(np.mean(latencies_ms)),
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "max": float(np.max(latencies_ms)),
            "n_samples": float(len(latencies_ms)),
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_full_evaluation(
    categories: list[str],
    *,
    data_root: Path,
    models_dir: Path,
    size: int,
    target_spatial: int,
    coreset_ratio: float,
    cross_category: bool,
) -> list[EvalResult]:
    """Train one detector per category, evaluate in-domain, and optionally
    cross-evaluate every (train, test) pair."""
    detectors: dict[str, PatchCoreDetector] = {}
    train_counts: dict[str, int] = {}
    results: list[EvalResult] = []

    models_dir.mkdir(parents=True, exist_ok=True)

    for cat in categories:
        cat_root = data_root / cat
        print(f"[train] {cat}")
        detector = train_detector(
            cat_root,
            size=size,
            target_spatial=target_spatial,
            coreset_ratio=coreset_ratio,
            model_version=f"mvtec-{cat}-v1",
        )
        detectors[cat] = detector
        train_counts[cat] = int(detector.metadata.n_training_images)
        bank_path = models_dir / f"image_{cat}.joblib"
        detector.save(bank_path)
        print(f"        bank: {detector.memory_bank.shape} -> {bank_path}")

        print(f"[eval ] in_domain {cat} -> {cat}")
        results.append(
            evaluate(
                detector,
                cat_root,
                train_category=cat,
                test_category=cat,
                n_train_images=train_counts[cat],
                size=size,
                experiment="in_domain",
            )
        )

    if cross_category and len(categories) >= 2:
        for train_cat in categories:
            for test_cat in categories:
                if train_cat == test_cat:
                    continue
                print(f"[eval ] cross    {train_cat} -> {test_cat}")
                results.append(
                    evaluate(
                        detectors[train_cat],
                        data_root / test_cat,
                        train_category=train_cat,
                        test_category=test_cat,
                        n_train_images=train_counts[train_cat],
                        size=size,
                        experiment="cross_category",
                    )
                )

    return results


def write_results_json(results: list[EvalResult], path: Path, *, meta: dict[str, Any]) -> None:
    payload = {
        "meta": meta,
        "experiments": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PatchCore on MVTec AD.")
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="category name (repeat for multiple). Default: bottle cable capsule",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/mvtec"), help="MVTec extraction root"
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="where memory banks go"
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("docs/evaluation/image-baseline.results.json"),
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--target-spatial", type=int, default=28)
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument(
        "--no-cross-category",
        action="store_true",
        help="skip cross-category experiments (in-domain only)",
    )
    args = parser.parse_args()

    categories = args.category or ["bottle", "cable", "capsule"]

    print(
        f"categories={categories}, size={args.size}, "
        f"target_spatial={args.target_spatial}, coreset_ratio={args.coreset_ratio}"
    )

    started = time.time()
    results = run_full_evaluation(
        categories,
        data_root=args.data_root,
        models_dir=args.models_dir,
        size=args.size,
        target_spatial=args.target_spatial,
        coreset_ratio=args.coreset_ratio,
        cross_category=not args.no_cross_category,
    )
    elapsed = time.time() - started

    meta = {
        "categories": categories,
        "size": args.size,
        "target_spatial": args.target_spatial,
        "coreset_ratio": args.coreset_ratio,
        "torch_version": torch.__version__,
        "wall_clock_seconds": round(elapsed, 1),
    }
    write_results_json(results, args.results_json, meta=meta)

    print(f"\nWrote {args.results_json}")
    for r in results:
        pix = f"{r.pixel_auroc:.4f}" if r.pixel_auroc is not None else "n/a"
        print(
            f"  {r.experiment:14s} {r.train_category:10s} -> {r.test_category:10s}  "
            f"img-AUROC={r.image_auroc:.4f}  pix-AUROC={pix}  "
            f"mean={r.latency_ms['mean']:.0f}ms  p95={r.latency_ms['p95']:.0f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
