"""Distribution-drift primitives.

A *foundation* for drift monitoring — pure numerical functions, no API
endpoint, no persistence. Phase 1.7 will wire these into a periodic drift
service. Keeping the scope tight here means the math stays simple to review
and the project does not collect half-finished plumbing.

Two complementary measures are exposed:

- **Kolmogorov-Smirnov statistic** (``compute_ks``): non-parametric, sensitive
  to differences anywhere in the distribution, returns a value in [0, 1].
- **Population Stability Index** (``compute_psi``): the industry-standard
  metric for tabular feature drift. Rule of thumb commonly used by credit-risk
  teams: ``< 0.1`` stable, ``0.1-0.25`` moderate shift, ``> 0.25`` strong shift.

Both functions handle multi-column feature matrices by computing the metric
per column and returning the *maximum*. The max (rather than mean) is chosen
because a single feature shifting catastrophically is operationally more
interesting than several features shifting slightly.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from scipy import stats

DEFAULT_PSI_BINS: Final[int] = 10
_EPSILON: Final[float] = 1e-6


def _validate_pair(reference: np.ndarray, current: np.ndarray) -> None:
    if reference.ndim != current.ndim:
        raise ValueError(
            f"reference and current must have same ndim; got {reference.ndim} vs {current.ndim}"
        )
    if reference.ndim == 2 and reference.shape[1] != current.shape[1]:
        raise ValueError(
            f"feature counts differ: reference has {reference.shape[1]}, "
            f"current has {current.shape[1]}"
        )
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("reference and current must both be non-empty")


def compute_ks(reference: np.ndarray, current: np.ndarray) -> float:
    """Worst-case KS statistic across feature columns.

    Returns a single float in [0, 1]: 0 means identical, 1 means disjoint.
    """
    _validate_pair(reference, current)

    if reference.ndim == 1:
        result: float = float(stats.ks_2samp(reference, current).statistic)
        return result

    per_column = [
        float(stats.ks_2samp(reference[:, i], current[:, i]).statistic)
        for i in range(reference.shape[1])
    ]
    return max(per_column)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = DEFAULT_PSI_BINS,
) -> float:
    """Worst-case Population Stability Index across feature columns.

    Reference bin edges are quantile-based, so the function is scale-invariant.
    """
    _validate_pair(reference, current)

    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")

    if reference.ndim == 1:
        return _psi_one_column(reference, current, bins=bins)

    per_column = [
        _psi_one_column(reference[:, i], current[:, i], bins=bins)
        for i in range(reference.shape[1])
    ]
    return max(per_column)


def _psi_one_column(reference: np.ndarray, current: np.ndarray, *, bins: int) -> float:
    """Compute PSI for one feature column."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    # Avoid zero-width bins on constant columns by perturbing endpoints.
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(len(reference), 1) + _EPSILON
    cur_pct = cur_counts / max(len(current), 1) + _EPSILON

    contributions: np.ndarray = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(contributions.sum())


def classify_psi(psi: float) -> str:
    """Bucket a PSI value into the conventional severity labels."""
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "strong"
