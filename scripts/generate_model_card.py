"""Generate a Markdown model card from a saved model artifact + sidecar.

The card is a *human-readable companion* to the JSON metadata: same facts,
spelled out in a form that a non-author reader can scan in 60 seconds. Run
after training to refresh ``docs/model-cards/baseline.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factory_anomaly.ml.detector import ModelMetadata

CARD_TEMPLATE = """# Model card — {model_name}

> Auto-generated from the metadata sidecar. **Re-run** ``scripts/generate_model_card.py``
> after any retraining; do not hand-edit the generated sections.

## Overview

- **Model name:** `{model_name}`
- **Model version:** `{model_version}`
- **Trained at:** {trained_at}
- **sklearn version:** `{sklearn_version}`
- **Python version:** `{python_version}`

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

- **Training data shape:** {training_data_shape[0]} rows x {training_data_shape[1]} features
- **Training data hash (sha256):** `{training_data_hash}`
- **Source:** synthetic stream from `factory_anomaly.data.synthetic.make_example_dataset`
  (sine base + injected spikes + a tail trend). Re-derivable from the seed
  embedded in the training script.

## Architecture

Isolation Forest (`sklearn.ensemble.IsolationForest`) over rolling-window
features (mean, std, min, max, peak-to-peak) computed by
`factory_anomaly.ml.features.make_rolling_features`.

Hyperparameters (verbatim from the artifact):

```json
{hyperparameters_json}
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
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("models/baseline.joblib.meta.json"),
        help="path to the metadata sidecar JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/model-cards/baseline.md"),
        help="output Markdown path",
    )
    args = parser.parse_args()

    if not args.meta.exists():
        raise SystemExit(
            f"metadata sidecar not found: {args.meta}\nRun scripts/train_baseline.py first."
        )

    meta = ModelMetadata.from_json(args.meta.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = CARD_TEMPLATE.format(
        model_name=meta.model_name,
        model_version=meta.model_version,
        trained_at=meta.trained_at,
        sklearn_version=meta.sklearn_version,
        python_version=meta.python_version,
        training_data_shape=meta.training_data_shape,
        training_data_hash=meta.training_data_hash,
        hyperparameters_json=json.dumps(meta.hyperparameters, indent=2, sort_keys=True),
    )
    args.out.write_text(body)
    print(f"Wrote {args.out} (model_version={meta.model_version})")


if __name__ == "__main__":
    main()
