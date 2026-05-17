# ADR-0006: Image anomaly modality — model, library, and streaming integration

- **Status:** Accepted
- **Date:** 2026-05-17
- **Deciders:** Naoya Higashitani

## Context

[ADR-0001](0001-domain-and-dataset.md) deferred image and audio data to Phase 3.
This ADR opens that work. The repo's narrative is **unsupervised anomaly
detection on live factory streams**; Phase 3 expands the *modality* without
breaking that narrative.

Three questions are coupled and have to be settled together before any code
lands:

1. **Model**: which detector
2. **Library**: scratch / framework / hybrid
3. **Streaming integration**: how does a per-image classifier fit a repo
   whose story is "live sensor stream"

This ADR settles all three, plus dataset licensing and deployment scope —
both of which would otherwise leak into implementation surprises.

## Decisions

### Model: PatchCore (CVPR 2022)

Memory-bank nearest-neighbour over locally-aware patch features from a
frozen ImageNet backbone.

- Per-category AUROC on MVTec AD ≈ 99% (vs ≈ 95% for PaDiM).
- Coreset subsampling (greedy k-centre) is a meaningful design feature in
  its own right — worth implementing, not just calling.
- Inference is `sklearn.neighbors.NearestNeighbors` at portfolio scale —
  no FAISS dependency needed.

Reference: Roth et al., *Towards Total Recall in Industrial Anomaly
Detection*, CVPR 2022 ([arxiv 2106.08265](https://arxiv.org/abs/2106.08265)).

### Library: hybrid (torchvision + own scoring)

- **Backbone**: `torchvision.models.resnet50` with `IMAGENET1K_V2` weights, frozen.
- **Feature extraction**: forward hooks on `layer2` and `layer3` (the
  locally-aware mid-level layers per the paper).
- **Coreset construction**: greedy k-centre subsampling, implemented in
  numpy.
- **NN scoring**: `sklearn.neighbors.NearestNeighbors` (k=1 for anomaly score,
  k=N for the re-weighting term).

Dependencies live in an **optional `image` dependency group** in
`pyproject.toml`. Default `uv sync` does not install torch. CI runs a
dedicated `image-quality` job that installs the group.

### Streaming integration: `image-ingester` + dedicated `image-anomaly-api` sidecar

Mirrors the [ADR-0004](0004-inference-trigger-final.md) sensor pattern.
A new `image-ingester` container periodically POSTs a frame to the
image-anomaly API; the API runs inference and returns score + heatmap.
This preserves the "live factory feed" narrative — frames are produced and
scored on a cadence, not on demand.

A separate `image-scorer` (analogue of the sensor `scorer`) is **not**
introduced: image inference is bundled into the ingest request because
unlike `/infer` for time-series, image scoring has no notion of "score the
latest" without specifying *which* image. Splitting it would add ceremony
without clarity.

Service runs on a **dedicated Docker image** (`Dockerfile.image`) so the
existing 5-service stack stays torch-free.

### Dataset: MVTec AD (primary), VisA (secondary)

- **MVTec AD** — industry benchmark. License: **CC BY-NC-SA 4.0
  (non-commercial research only)**. Data is **not committed**; downloaded
  on demand by `scripts/download_mvtec.py`. **Trained memory banks are also
  not committed** — they are a derivative of the data and inherit the
  restriction.
- **VisA (Amazon)** — CC-BY-4.0, permissive. Used for cross-dataset
  evaluation in the honest-finding report.

### Deployment: built, not auto-deployed

The image-anomaly service is `docker-compose up`-able locally and a
Terraform module is shipped, but `deploy-aws.yml` does not deploy it by
default. App Runner overhead × idle cost would compromise the Phase 2
$0/month cost ceiling. Apply is opt-in via a manual workflow.

## Honest negative findings to be documented

Per the project's standing convention, Phase 3 surfaces real limitations
even when the headline numbers are good:

1. **Cross-category generalisation failure** — a memory bank trained on
   `bottle` evaluated against `cable` collapses. Demonstrates the
   per-category-bank constraint quantitatively.
2. **CPU inference latency asymmetry** — image inference takes
   500ms–2s/frame on CPU; the sensor stream side handles sub-100ms. The
   "streaming" framing only holds at low frame rate.

Both go into `docs/evaluation/image-baseline.md` alongside the per-category
AUROC table.

## Alternatives considered

| Option | Why rejected |
|---|---|
| **PaDiM** | Older (ICPR 2020), lower AUROC; per-pixel Gaussian + covariance matrices are heavier to ship and weaker on results. |
| **DRAEM / PatchSVDD** | Heavier training, no clear portfolio gain over PatchCore. |
| **`anomalib` (Intel, Apache-2.0) full integration** | Works in one import, but reduces Phase 3 to "wrote a Dockerfile around someone else's library". The portfolio signal of this phase is implementing the coreset + the feature-hook pipeline, not framework integration. |
| **Pure scratch including the backbone** | Re-implementing ResNet50 is anti-portfolio: everyone has done it and no reviewer doubts it works. |
| **"Grad-CAM-style" heatmap** (initially considered) | Technically wrong: Grad-CAM is a gradient-saliency method for *classifiers*. PatchCore produces a native per-patch anomaly map. Renamed accordingly. |
| **File-upload `/classify-image` endpoint only** | Breaks the streaming narrative. Replaced with `image-ingester` sidecar. |
| **Deploy image service alongside Phase 2 AWS stack** | Breaks the Phase 2 cost ceiling. Made opt-in. |

## Consequences

- ~400 new LOC: feature extractor, coreset, scorer, FastAPI service,
  ingester sidecar.
- New `Dockerfile.image` (~1.5GB target: torch + torchvision + baked
  ResNet50 weights). Existing `Dockerfile` unchanged.
- New CI job `image-quality` using `uv sync --group image`. Existing
  `quality` job unaffected — default install stays light.
- Coverage gate (95%) maintained — torch glue tested via shape contracts
  with small random fixtures; coreset and NN scoring are pure numpy /
  sklearn and fully unit-testable.
- New evaluation report `docs/evaluation/image-baseline.md` covering
  per-category MVTec AUROC, the two honest negative findings above, and
  CPU latency measurements.
