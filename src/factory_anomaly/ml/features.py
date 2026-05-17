"""Rolling-window feature extraction.

The detector works on *feature vectors*, not raw samples. For a univariate
stream, each window contributes one row with a handful of summary statistics
(mean, std, min, max, peak-to-peak). This is the simplest representation that
gives Isolation Forest enough signal on the kinds of anomalies generated in
``factory_anomaly.data.synthetic``.

Multivariate support: ``make_multivariate_rolling_features`` stacks per-channel
features column-wise so a single Isolation Forest can see all sensors at once.
Used in Phase 1.8 to address the binding constraint identified by Phase 1.7's
SKAB evaluation (single-channel scoring was the floor, not periodicity).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

FEATURE_NAMES: Final[tuple[str, ...]] = ("mean", "std", "min", "max", "p2p")


def make_rolling_features(values: np.ndarray, window: int) -> pd.DataFrame:
    """Compute rolling-window features over a 1-D signal.

    Returns a DataFrame with one row per *complete* window — the first
    ``window - 1`` samples are dropped. The output index matches the position
    of the **right edge** of each window in the input array, so callers can
    align feature rows back to original timestamps without extra bookkeeping.
    """
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")
    if window <= 1:
        raise ValueError(f"window must be > 1, got {window}")
    if len(values) < window:
        raise ValueError(f"signal has {len(values)} samples, need at least window={window}")

    series = pd.Series(values)
    roll = series.rolling(window=window, min_periods=window)

    features = pd.DataFrame(
        {
            "mean": roll.mean(),
            "std": roll.std(ddof=0),
            "min": roll.min(),
            "max": roll.max(),
        }
    ).dropna()
    features["p2p"] = features["max"] - features["min"]
    return features[list(FEATURE_NAMES)]


def make_multivariate_rolling_features(
    channels: dict[str, np.ndarray], window: int
) -> pd.DataFrame:
    """Stack per-channel rolling features column-wise.

    All channels must have the same length so the output rows align to the
    same right-edge timestamps. Column names follow ``<channel>__<feature>``
    (double underscore) so column-name parsing stays unambiguous even when
    sensor names contain underscores.
    """
    if not channels:
        raise ValueError("at least one channel required")

    lengths = {len(v) for v in channels.values()}
    if len(lengths) > 1:
        raise ValueError(f"all channels must have the same length; got {sorted(lengths)}")

    blocks: list[pd.DataFrame] = []
    for name, values in channels.items():
        block = make_rolling_features(values, window=window)
        block.columns = [f"{name}__{c}" for c in block.columns]
        blocks.append(block)

    return pd.concat(blocks, axis=1)
