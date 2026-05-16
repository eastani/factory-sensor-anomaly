"""Render a static preview image of the dashboard's main chart.

We deliberately avoid headless-browser screenshot tools (Playwright /
Selenium) because they add hundreds of MB of dependencies for a single PNG.
Instead, this script generates the *same* plot shapes the live dashboard
would render — from the same synthetic data — using matplotlib. The result
goes in ``docs/assets/dashboard-preview.png`` and is referenced from the
README so a casual reader sees the project's output without having to run
docker compose first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factory_anomaly.data import make_example_dataset
from factory_anomaly.ml import AnomalyDetector
from factory_anomaly.ml.features import make_rolling_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/dashboard-preview.png"),
        help="output PNG path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    sig = make_example_dataset(seed=args.seed)
    features = make_rolling_features(sig.values, window=args.window).to_numpy()
    detector = AnomalyDetector(model_version="preview")
    detector.fit(features)
    scores = detector.score(features)
    preds = detector.predict(features)

    # Right-edge alignment of features to original samples.
    timestamps = pd.to_datetime(sig.timestamps)
    aligned_ts = timestamps[args.window - 1 :]
    aligned_values = sig.values[args.window - 1 :]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(11, 5.5),
        gridspec_kw={"height_ratios": [3, 2]},
        sharex=True,
    )

    # Top: sensor stream + anomaly markers.
    ax_top.plot(aligned_ts, aligned_values, color="#1f77b4", linewidth=1.1, label="sensor")
    anomaly_idx = np.where(preds)[0]
    if len(anomaly_idx):
        ax_top.scatter(
            aligned_ts[anomaly_idx],
            aligned_values[anomaly_idx],
            color="#d62728",
            marker="x",
            s=50,
            linewidths=1.6,
            label="anomaly",
            zorder=5,
        )
    ax_top.set_title("Sensor stream — factory-sensor-anomaly dashboard preview")
    ax_top.set_ylabel("value")
    ax_top.legend(loc="upper left", frameon=False)
    ax_top.grid(alpha=0.2)

    # Bottom: anomaly score history.
    ax_bottom.plot(aligned_ts, scores, color="#2ca02c", linewidth=1.0)
    ax_bottom.axhline(0, color="#888", linestyle="--", linewidth=0.6)
    ax_bottom.set_ylabel("anomaly score\n(higher = more anomalous)")
    ax_bottom.set_xlabel("time")
    ax_bottom.grid(alpha=0.2)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
