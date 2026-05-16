"""Train the baseline Isolation Forest on synthetic example data.

Run once before bringing up the API so there is a model artifact to load:

    uv run python scripts/train_baseline.py

Re-running produces a fresh model_version derived from the current timestamp.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from factory_anomaly.data import make_example_dataset
from factory_anomaly.ml import AnomalyDetector
from factory_anomaly.ml.features import make_rolling_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("models/baseline.joblib"), help="output path"
    )
    parser.add_argument("--window", type=int, default=20, help="rolling window size")
    parser.add_argument("--seed", type=int, default=42, help="random seed for synthetic data")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="model_version label (default: UTC timestamp)",
    )
    parser.add_argument(
        "--contamination",
        default="auto",
        help="IsolationForest contamination (float or 'auto')",
    )
    args = parser.parse_args()

    contamination: float | str
    try:
        contamination = float(args.contamination)
    except ValueError:
        contamination = args.contamination

    version = args.version or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    sig = make_example_dataset(seed=args.seed)
    features = make_rolling_features(sig.values, window=args.window).to_numpy()

    detector = AnomalyDetector(model_version=version, contamination=contamination)
    detector.fit(features)
    written = detector.save(args.out)

    print(f"Trained baseline model -> {written}")
    print(f"  version:        {version}")
    print(f"  samples seen:   {features.shape[0]}")
    print(f"  features/row:   {features.shape[1]}")
    print(f"  contamination:  {contamination}")


if __name__ == "__main__":
    main()
