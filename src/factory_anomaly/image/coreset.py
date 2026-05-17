"""Greedy k-centre coreset subsampling.

PatchCore's memory bank is the concatenation of *every* patch descriptor
from every training image — typically millions of vectors per category.
Storing all of them defeats the deployment story: the bank has to fit in
the inference container's memory and ship as part of the model artifact.

The paper's solution is **greedy k-centre subsampling**: pick a subset
``M ≪ N`` whose maximum coverage radius (distance from any input point
to its nearest selected point) is small. Greedy picks each new point as
the one currently farthest from the already-selected set. This is the
NP-hard k-centre problem; the greedy algorithm is a 2-approximation
([Gonzalez 1985](https://www.sciencedirect.com/science/article/pii/0304397585902245)).

The implementation here is **exact greedy**, not the approximate
sparse-projection variant the paper uses for very large banks. At
portfolio scale (~10^5 patches per category) exact is fast enough and
removes one source of "is this implemented correctly?" noise.
"""

from __future__ import annotations

import numpy as np


def greedy_coreset(
    features: np.ndarray,
    n_samples: int,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Return indices into ``features`` selected by greedy k-centre.

    Parameters
    ----------
    features
        ``(N, D)`` float array.
    n_samples
        Number of points to select. Must be ``>= 1``. If ``>= N``, all
        indices are returned in original order (no subsampling).
    random_state
        Seed for the initial random pick. Subsequent picks are deterministic.

    Returns
    -------
    np.ndarray
        ``(n_samples,)`` int array of indices into ``features``.

    Notes
    -----
    Cost is ``O(N * n_samples * D)`` for distance evaluations plus
    ``O(N * n_samples)`` for the running ``min_dist`` buffer. For 100k x
    512-D features and 10% coreset (~10k samples) this is a few seconds
    on CPU, which is the regime PatchCore targets.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    n_total = features.shape[0]
    if n_samples >= n_total:
        return np.arange(n_total)

    rng = np.random.default_rng(random_state)
    first = int(rng.integers(n_total))

    selected = np.empty(n_samples, dtype=np.int64)
    selected[0] = first

    # Distance from every input point to the *nearest* selected point so far.
    # Initialise to distance from the first pick; update incrementally as
    # each new point is added — avoids recomputing against the full set.
    min_dist = np.linalg.norm(features - features[first], axis=1)

    for i in range(1, n_samples):
        next_idx = int(np.argmax(min_dist))
        selected[i] = next_idx
        new_dist = np.linalg.norm(features - features[next_idx], axis=1)
        np.minimum(min_dist, new_dist, out=min_dist)

    return selected
