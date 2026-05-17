# Architecture Decision Records

Lightweight ADRs in the [MADR](https://adr.github.io/madr/) style. Each captures **one** decision with context, alternatives, and consequences. Numbered, append-only — supersede rather than edit.

| # | Title | Status |
|---|-------|--------|
| [0001](0001-domain-and-dataset.md) | Domain selection & primary dataset | Accepted |
| [0002](0002-tech-stack.md) | Core tech stack & ML approach | Accepted |
| [0003](0003-inference-trigger.md) | Inference trigger strategy | Superseded by 0004 |
| [0004](0004-inference-trigger-final.md) | Inference trigger — final decision (ingester + scorer sidecars) | Accepted |
| [0005](0005-cloud-architecture.md) | Cloud architecture (AWS primary, Azure secondary) | Accepted |
| [0006](0006-image-anomaly-modality.md) | Image anomaly modality — PatchCore, hybrid library, sidecar integration | Accepted |
