# Model card — isolation_forest

> Auto-generated from the metadata sidecar. **Re-run** ``scripts/generate_model_card.py``
> after any retraining; do not hand-edit the generated sections.

## Overview

- **Model name:** `isolation_forest`
- **Model version:** `20260516-064837`
- **Trained at:** 2026-05-16T06:48:38.003803+00:00
- **sklearn version:** `1.8.0`
- **Python version:** `3.12.8`

## Intended use

Unsupervised anomaly scoring for factory-sensor time-series streams
(pressure, temperature, RPM, vibration). Designed for the streaming
inference pattern used by the API's `/infer` endpoint — one feature row
per call, score returned in milliseconds.

## Out-of-scope

- Multi-machine joint inference (model is per-signal).
- Forecasting / remaining-useful-life regression — see the companion repo
  [predictive-maintenance-cmapss](https://github.com/eastani/predictive-maintenance-cmapss).
- Image / acoustic modalities (Phase 3 work).

## Training data

- **Training data shape:** 981 rows x 5 features
- **Training data hash (sha256):** `8c0204637384aaf7aac168c7e4005733bcf973dd8ea9ae7a98bdb4ebef941695`
- **Source:** synthetic stream from `factory_anomaly.data.synthetic.make_example_dataset`
  (sine base + injected spikes + a tail trend). Re-derivable from the seed
  embedded in the training script.

## Architecture

Isolation Forest (`sklearn.ensemble.IsolationForest`) over rolling-window
features (mean, std, min, max, peak-to-peak) computed by
`factory_anomaly.ml.features.make_rolling_features`.

Hyperparameters (verbatim from the artifact):

```json
{
  "contamination": "auto",
  "n_estimators": 100,
  "random_state": 42
}
```

## Evaluation

See `docs/evaluation/baseline-skab.md` for an evaluation on real SKAB data
(rotating-pump testbed). The synthetic training data is too easy to score
fairly on; SKAB is the honest test.

## Limitations

- **Periodicity confound:** raw Isolation Forest tends to flag periodic
  peaks as anomalies and to miss anomalies that coincide with peaks. The
  STL-decomposition wrapper promised in ADR-0002 is *not yet implemented* —
  scheduled for Phase 1.7. Until then, expect false positives on strongly
  cyclic signals.
- **Single-channel:** features are computed per sensor. Cross-sensor
  correlation information is unused.
- **No drift gate:** see `factory_anomaly.ml.drift` for the foundation
  functions — the wiring into a periodic drift service is a Phase 1.7 task.

## Reproducing

```bash
uv run python scripts/train_baseline.py --version <yourtag>
uv run python scripts/generate_model_card.py --version <yourtag>
```
