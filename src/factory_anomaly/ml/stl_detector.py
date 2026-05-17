"""Isolation Forest on STL residuals.

Closes the implementation gap recorded in [ADR-0002](../../docs/adr/0002-tech-stack.md):
the "improvement over baseline IF" was promised but not yet implemented.

Why this matters
----------------

Raw Isolation Forest tends to flag periodic peaks (shift changes, cycle
boundaries, scheduled load fluctuations) as anomalies, and to miss real
anomalies that happen to coincide with those peaks. STL (Seasonal-Trend
decomposition using Loess) separates a 1-D signal into ``trend + seasonal +
residual`` components; running IF on the *residual* removes the periodic
confound and lets the model focus on genuine deviations.

Pattern adopted from [Jacer7/Anomaly_Detection](https://github.com/Jacer7/Anomaly_Detection)
and the periodicity-confound write-up referenced in ADR-0002.

Scope
-----

Phase 1.7 wires this into the **batch evaluation** path
(``scripts/evaluate_skab.py``) so the SKAB result can be re-measured against
the baseline. Streaming integration with ``/infer`` is deferred to Phase 1.8
because STL needs a full multi-period window for stable decomposition, which
the current single-window-per-call API contract does not expose.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.tsa.seasonal import STL

from factory_anomaly.ml.detector import AnomalyDetector, ModelMetadata
from factory_anomaly.ml.features import make_rolling_features


class StlAnomalyDetector:
    """Higher-level detector: STL residual extraction + rolling features + IF.

    Operates on raw 1-D time-series values. The internal ``AnomalyDetector``
    instance handles the IF logic + persistence; this class adds the
    decomposition step in front.
    """

    def __init__(
        self,
        model_version: str,
        *,
        period: int = 50,
        window: int = 20,
        robust: bool = True,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 42,
    ) -> None:
        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}")
        if window <= 1:
            raise ValueError(f"window must be > 1, got {window}")

        self.period = period
        self.window = window
        self.robust = robust
        self._inner = AnomalyDetector(
            model_version=model_version,
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    @property
    def model_version(self) -> str:
        return self._inner.model_version

    @property
    def metadata(self) -> ModelMetadata:
        return self._inner.metadata

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    def _residuals(self, values: np.ndarray) -> np.ndarray:
        """Return the STL residual component for a 1-D signal.

        ``values`` must contain at least ``2 * period`` samples for STL to
        return a stable decomposition; we let statsmodels raise its own error
        otherwise rather than silently truncating.
        """
        if values.ndim != 1:
            raise ValueError(f"values must be 1-D, got shape {values.shape}")
        result = STL(values, period=self.period, robust=self.robust).fit()
        return np.asarray(result.resid, dtype=float)

    def _features_from_values(self, values: np.ndarray) -> np.ndarray:
        residuals = self._residuals(values)
        return make_rolling_features(residuals, window=self.window).to_numpy()

    def fit(self, values: np.ndarray) -> StlAnomalyDetector:
        features = self._features_from_values(values)
        self._inner.fit(features)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        """Per-window anomaly scores aligned to right edges of the rolling window."""
        features = self._features_from_values(values)
        return self._inner.score(features)

    def predict(self, values: np.ndarray) -> np.ndarray:
        features = self._features_from_values(values)
        return self._inner.predict(features)

    def hyperparameters(self) -> dict[str, Any]:
        """Surfaceable parameters — useful for evaluation reports."""
        return {
            "period": self.period,
            "window": self.window,
            "robust": self.robust,
            **self._inner._estimator.get_params(),
        }
