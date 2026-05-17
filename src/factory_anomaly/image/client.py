"""HTTP client for the image-anomaly service.

Lives in the image package so the Streamlit dashboard (Phase 3.4) can
import it directly without dragging torch in — this module depends on
``httpx`` only, not on PyTorch. The dashboard container therefore stays
torch-free, which is the load-bearing assumption from ADR-0006.

Errors are normalised to ``ImageApiClientError`` so callers do not have to
import ``httpx`` themselves.
"""

from __future__ import annotations

from typing import Any, cast

import httpx


class ImageApiClientError(RuntimeError):
    """Raised when the image API returns a non-success status or is unreachable."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ImageApiClient:
    """Thin wrapper around the image-anomaly REST API."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        # Default timeout is higher than the sensor API client's 5s because
        # CPU image inference can legitimately take 1-2 seconds end-to-end.
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ImageApiClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def base_url(self) -> str:
        return self._base_url

    def healthz(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._get("/healthz"))

    def predict(self, image_bytes: bytes, *, filename: str = "image.png") -> dict[str, Any]:
        """POST an image (PNG/JPEG bytes) for anomaly scoring."""
        files = {"image": (filename, image_bytes, "application/octet-stream")}
        try:
            response = self._client.post("/image/predict", files=files)
        except httpx.HTTPError as exc:
            raise ImageApiClientError(f"POST /image/predict failed: {exc}") from exc

        if response.status_code >= 400:
            raise ImageApiClientError(
                f"POST /image/predict -> HTTP {response.status_code}: {response.text}",
                status_code=response.status_code,
            )
        return cast(dict[str, Any], response.json())

    def _get(self, path: str) -> Any:
        try:
            response = self._client.get(path)
        except httpx.HTTPError as exc:
            raise ImageApiClientError(f"GET {path} failed: {exc}") from exc

        if response.status_code >= 400:
            raise ImageApiClientError(
                f"GET {path} -> HTTP {response.status_code}: {response.text}",
                status_code=response.status_code,
            )
        return response.json()
