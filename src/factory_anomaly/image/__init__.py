"""Image anomaly module — PatchCore detector.

See [ADR-0006](../../../docs/adr/0006-image-anomaly-modality.md) for the
design rationale (model choice, hybrid-library decision, streaming bridge).

Two import surfaces:

- **Package-level (this file)** — re-exports only the **torch-free** pieces:
  the HTTP client and the (pure-numpy) coreset helper. The dashboard
  container imports from here and must stay torch-free per ADR-0006.
- **Submodule** — ``factory_anomaly.image.detector``,
  ``factory_anomaly.image.feature_extractor``, ``factory_anomaly.image.api``,
  and ``factory_anomaly.image.workers`` require the optional ``image``
  dependency group (``uv sync --group image``). Import them directly when
  you need them.

If you import the heavy submodules from a context that lacks torch, you
will see ``ModuleNotFoundError: No module named 'torch'``. That is the
intended behaviour — failing fast at import time beats a half-loaded
service.
"""

from factory_anomaly.image.client import ImageApiClient, ImageApiClientError
from factory_anomaly.image.coreset import greedy_coreset

__all__ = [
    "ImageApiClient",
    "ImageApiClientError",
    "greedy_coreset",
]
