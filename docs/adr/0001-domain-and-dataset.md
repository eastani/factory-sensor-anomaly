# ADR-0001: Domain selection & primary dataset

- **Status:** Accepted
- **Date:** 2026-05-16
- **Deciders:** Naoya Higashitani

## Context

This is a portfolio project to demonstrate end-to-end ML engineering capability (data → model → API → UI → cloud). The target domain is **factory sensor data** — pressure, temperature, rotation speed (RPM), and vibration signals. The chosen dataset must satisfy:

1. **Publicly downloadable**, with a permissive enough license to redistribute analysis (not necessarily the data).
2. **Time-series structure** with multiple sensor channels — matches the "real industrial telemetry" narrative.
3. **Labelled anomalies** — even though the model will be unsupervised, labels are needed to score it.
4. **Tabular-friendly** — Phase 1 uses scikit-learn (Isolation Forest, LOF), not deep models. Image / raw audio data is deferred to Phase 3.

## Decision

Use **two datasets**, primary + secondary.

### Primary: [Kaggle Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)
- 52 numerical sensor channels, minute-level timestamps, ~220k rows.
- Label column `machine_status` with three classes: `NORMAL`, `BROKEN`, `RECOVERING`.
- Kaggle open dataset (registration required to download — acceptable; downloaded once locally, never committed).

### Secondary: [SKAB — Skoltech Anomaly Benchmark](https://github.com/waico/SKAB)
- Multi-sensor industrial water-circulation testbed (rotating pump + tank): accelerometer, current, pressure, temperature, vibration on multiple axes.
- Anomaly **intervals** labelled by domain experts (35 labelled anomaly events across 34 CSV files).
- Hosted on GitHub directly — no registration, no manual download — which means it can be pulled from CI for integration tests.
- License is GPL-3.0 on the data; analysis code in this repository remains MIT and does not redistribute the data.
- Used in Phase 1.5+ to validate the model on a *second*, independent factory-style stream. The secondary dataset is deliberately a different machine type (water circulation vs. liquid pump) to test that the architecture is schema-agnostic.

## Alternatives considered

| Option | Why not (now) |
|--------|---------------|
| [NASA IMS Bearings](https://data.nasa.gov/dataset/ims-bearings) | Strong dataset, but the companion repo [predictive-maintenance-cmapss](https://github.com/eastani/predictive-maintenance-cmapss) already uses a NASA rotating-machinery dataset (CMAPSS turbofan). Two "NASA + rotating equipment" projects on the same GitHub profile would dilute the narrative. SKAB gives an independent, industrially-realistic source. |
| [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) | Well-labelled but a [benchmark study by Smith & Randall (2015)](https://www.sciencedirect.com/science/article/abs/pii/S0888327015002034) shows some fault sizes are trivially separable; reporting 99%+ accuracy on CWRU is not a credible result. Reserved as a possible Phase 3 add-on with the harder 0.021"+ fault sizes. |
| [MIMII (Hitachi sound)](https://zenodo.org/records/3384388) | Audio modality — strong dataset but does not match the "pressure/temp/RPM" framing for Phase 1. Candidate for Phase 3 multi-modal expansion. |
| [UCI SECOM](https://archive.ics.uci.edu/ml/datasets/SECOM) | Severe 1:14 class imbalance + 591 features + heavy missingness — better as a *classification* demo than time-series anomaly detection. |
| [Numenta NAB](https://github.com/numenta/NAB) | Curated benchmark but the time-series are heterogeneous (AWS, Twitter, traffic) — does not tell a factory story. Will reuse the **NAB scoring methodology** even though we don't use the data. |

## Consequences

**Positive**
- Pump dataset is tabular, labelled, time-indexed — Phase 1 can ship without feature-engineering detours.
- Two datasets force the architecture to be dataset-agnostic from day one (no hardcoded schema).
- Public, citable sources make the project credible.

**Negative**
- Pump dataset's `RECOVERING` class is genuinely ambiguous. Naive tutorials either drop it or merge into `NORMAL`, both of which inflate metrics. This project will **report results for both label collapsings** and explain the tradeoff in the model card.
- Kaggle registration is a manual friction point for a first-time cloner. Solution: provide a synthetic data generator (sine + spike) as a zero-setup fallback for local development.
- SKAB is GPL-3.0 on the **data**; the analysis code in this repository remains MIT. The data is not redistributed — `scripts/download_data.sh` will fetch it directly from `waico/SKAB` at install time.

## Open questions

- Will the `RECOVERING` class be modelled at all, or only used for evaluation?
- For Phase 1, do we ingest the Kaggle CSV in bulk, or replay it as a synthetic stream to demonstrate the "live ingest" architecture? (Leaning toward **replay** — it makes the Streamlit dashboard meaningful.)
