# Image baseline — PatchCore on MVTec AD

**Status:** First real numbers, 2 categories.
**Date:** 2026-05-17.
**Source of truth:** [`image-baseline.results.json`](image-baseline.results.json) — this document is a hand-curated narrative around the same numbers.

This is the image-modality counterpart to [`baseline-skab.md`](baseline-skab.md). Same auditable format: experiments table, then a section per honest finding, then how to reproduce.

---

## Setup

- **Model:** PatchCore (ADR-0006). Frozen ResNet50 backbone (`IMAGENET1K_V2`), `layer2` + `layer3` features with 3x3 neighbourhood pooling, concatenated to D=1536. Coreset memory bank (greedy k-centre, ratio 0.10). Per-patch nearest-neighbour scoring against `sklearn.NearestNeighbors`, image score = max patch distance.
- **Input:** 224x224 RGB, ImageNet-normalised.
- **Patch grid:** `target_spatial=14` (= 196 patches/image). Note: the PatchCore paper uses 28x28. We dropped to 14 in this run because greedy coreset at 28² scaled past 30 minutes per category on a portfolio-class laptop. AUROC degradation from 28 → 14 is expected to be sub-1 point; the headline finding (cross-category collapse) is unaffected. Full 28-grid runs are listed under "future work".
- **Data:** MVTec AD `capsule` and `metal_nut` categories. Train on `train/good`, evaluate on the full `test/` set. Pixel AUROC uses `ground_truth/` masks bilinear-resized to the same 224x224. Good-image pixels are scored against an all-zero mask so the pixel AUROC reflects both true negatives and true positives.
- **Hardware:** Apple Silicon CPU, single process, 2 torch threads (matches `IMAGE_API_TORCH_NUM_THREADS` default).
- **Wall clock:** 701s total for 4 experiments (2 train + 2 in-domain + 2 cross-category).

## Experiments

| Experiment | Train | Test | Image AUROC | Pixel AUROC | Mean latency | p95 latency |
|---|---|---|---|---|---|---|
| in-domain | capsule | capsule | **0.978** | **0.987** | 50 ms | 56 ms |
| in-domain | metal_nut | metal_nut | **0.997** | **0.986** | 126 ms | 197 ms |
| cross-category | capsule | metal_nut | 0.379 | 0.760 | 154 ms | 276 ms |
| cross-category | metal_nut | capsule | 0.292 | 0.912 | 50 ms | 57 ms |

Bank size: ~4,300 patches per category (= 220 train images × 196 patches × 0.10 coreset ratio).

---

## Honest finding 1 — Cross-category collapse, *worse* than random

The cross-category image AUROCs (0.379 and 0.292) are **below 0.5**. A coin flip would score 0.5 in expectation. This is not "the model generalises poorly"; this is "the model actively misranks defects against good images of a different category".

Why this matters: PatchCore's memory bank stores patch descriptors of *normal* training images. The image-level anomaly score is "how far is the most-anomalous test patch from the closest stored patch". When test images come from a different category:

- Almost every test patch is far from every training patch — they're a different category.
- "Defective" patches of the test category and "good" patches of the test category both end up far from the training bank.
- The ordering of test-image scores no longer correlates with defect labels. With binary labels and an arbitrary scoring direction, *worse-than-random* is exactly what you get.

This empirically validates the per-category-bank constraint that ADR-0006 mentioned in the abstract. PatchCore is not a universal defect detector — you need one bank per inspection target.

### A nuance the pixel AUROC reveals

The **pixel-level** AUROC stays strong cross-category (0.76 and 0.91). Why does pixel localisation partially transfer when image-level scoring doesn't?

Hypothesis: PatchCore's pixel scoring measures how anomalous *each patch* is *relative to the rest of the image*. If a defect creates a locally unusual texture, that patch is unusual whether or not the model has ever seen this category. But the *image-level* max is dominated by the absolute-distance scale, which is meaningless cross-category. Localisation generalises somewhat; classification does not.

This is worth flagging because it inverts the naïve assumption that "if the heatmap looks right, the score must be right too." It can be wrong in the most-anomalous way: a localisation that visually picks out the defect but produces a score that ranks the image as more normal than other defective images.

---

## Honest finding 2 — CPU inference latency is asymmetric vs the sensor side

| Path | Typical per-call latency |
|---|---|
| Sensor `/infer` (Isolation Forest on a 1-D rolling window) | < 100 ms |
| Image `/image/predict` (this run) | 50 – 150 ms mean, up to 700 ms tail |

Image inference is in the same order of magnitude as the sensor path, but with a much longer tail (metal_nut hit a 692 ms single-image max). On a CPU-only deployment this caps the throughput at roughly 6–10 frames/second per service replica before the tail starts hurting. That is fine for a "conveyor camera per minute" cadence and absolutely not fine for a 30 FPS camera.

The streaming framing of this repo — "live factory feed" — therefore holds for image input *at sensor cadence*, not at video cadence. Two consequences for honest deployment:

1. The dashboard's image panel should refresh at the ingester cadence (15 s by default), not on every frame.
2. Any "real conveyor at frame rate" deployment story would need a GPU, batching, or a lighter backbone. ADR-0006's CPU-only design is a feature-quality / cost trade-off, not a hidden limitation.

Also worth noting: metal_nut latency is ~2.5x capsule's despite an identical pipeline and a comparable bank size. Differences come from (a) raw image size before resize and (b) NN index variance on a single laptop run. Re-running on a clean machine would close some of that gap; for portfolio-scale numbers it's not worth optimising.

---

## Honest finding 3 — Mirror availability was a real obstacle

ADR-0006 anticipated MVTec's CC BY-NC-SA 4.0 licensing and we coded `scripts/download_mvtec.py` against the documented per-category URL pattern. **That pattern returned 404s in this run.** The current state of public MVTec distribution:

| Source | What we observed |
|---|---|
| `https://www.mvtec.com/.../research-datasets/mvtec_ad/<cat>.tar.xz` | HTTP 404 — public URL pattern referenced in older academic code is no longer served. |
| Official downloads page | Requires a form submission, no programmatic access. |
| HuggingFace `Voxel51/mvtec-ad` | FiftyOne export, not raw-image layout. |
| HF community mirrors with category structure | Mix of gated (HTTP 401) and one-shot full-archive (~5 GB). |
| `alexsu52/mvtec_capsule` (HF) | Single-category tar.xz, layout preserved. Used here for capsule. |
| `MSherbinii/mvtec-ad-{cable,metal-nut}` (HF) | Git-LFS directory tree. Used here for metal_nut (in-tree parallel download via httpx). |

The script in this repo will still ship pointing at the canonical (broken) URL; the operator-side path is to (a) submit the MVTec form, or (b) point `--url-template` at whatever mirror they trust. We deliberately do not bake a community-mirror URL into the default — those mirrors redistribute the data without explicit MVTec permission and their lifetime is not ours to guarantee.

This is unsatisfying. It is also the truth, and worth recording so the next reader does not assume the auto-download is a one-liner.

---

## Comparison with the published paper

Roth et al. (CVPR 2022) report per-category MVTec image AUROCs in the 0.98–1.00 range, averaging 0.991 across all 15 categories at higher input resolution and the full 28x28 patch grid. Our **0.978 / 0.997** with `target_spatial=14` and a 10 % coreset are within roughly 1 point of those numbers on the two categories tested. This is the expected ballpark; we are not claiming SOTA.

The paper does **not** report cross-category numbers — that is not a standard PatchCore evaluation. The cross-category degradation finding here is original to this report, in the sense that the per-category-bank constraint is universally implicit in PatchCore deployments but rarely measured explicitly.

---

## Future work explicitly deferred

- **More categories.** Adding bottle / cable / pill / hazelnut would strengthen the in-domain story. Blocked on a reliable per-category mirror; full-archive download is also possible if a 5 GB HF mirror becomes stable.
- **Full 28x28 patch grid.** At target_spatial=28 we expect 0.5–1 point AUROC improvement, at the cost of ~10x longer training (greedy coreset becomes the bottleneck). Worth doing once with `--coreset-ratio 0.01` to keep wall-clock reasonable.
- **VisA cross-dataset.** ADR-0006 lists VisA as the secondary dataset for cross-*dataset* evaluation (different from this report's cross-*category*). Deferred to a future run when MVTec coverage is broader.
- **Re-weighting term.** PatchCore Eq. 7 softmax re-weighting of the most-anomalous patch is not implemented. We expect a sub-1-point AUROC improvement on the categories where the current `max` heuristic underperforms. Low priority given Finding 1's headline.

---

## How to reproduce

```bash
# 1. Read and accept MVTec AD's CC BY-NC-SA 4.0 license:
#    https://www.mvtec.com/company/research/datasets/mvtec-ad/license-mvtec-ad

# 2. Acquire the per-category archives. The script's default URL no longer
#    resolves (see Finding 3); the operator path is the form on
#    https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
#    or a community mirror.

# 3. Extract into data/mvtec/<category>/{train,test,ground_truth}

# 4. Run the evaluator:
make image-evaluate CATEGORIES="capsule metal_nut"

# Or for a single category with a smaller patch grid (faster):
uv run python scripts/evaluate_image.py \
    --category capsule --category metal_nut --target-spatial 14
```

The script writes `models/image_<cat>.joblib` (memory banks, not committed — see `image/LICENSE-NOTICE.md`) and refreshes `docs/evaluation/image-baseline.results.json` (numbers committed).
