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
  peaks as anomalies. The STL-residual variant promised in ADR-0002 was
  *implemented* in Phase 1.7 (`StlAnomalyDetector`) and is the right
  default on signals with strong cyclic structure (the synthetic preview
  in the README is one such case). Honest evaluation on SKAB showed it
  does not improve over the raw IF on that dataset — signal selection
  mattered more there. See `docs/evaluation/baseline-skab.md`.
- **Single-channel:** features are computed per sensor. Cross-sensor
  correlation information is unused — slated for Phase 1.8 (multivariate
  features stacking accelerometer + current + pressure).
- **No drift gate:** see `factory_anomaly.ml.drift` for the foundation
  functions — wiring into a periodic drift service is a future task.

## Reproducing

```bash
uv run python scripts/train_baseline.py --version <yourtag>
uv run python scripts/generate_model_card.py --version <yourtag>
```
