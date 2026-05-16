"""Typed HTTP client for the FastAPI backend.

The Streamlit view in ``dashboard/app.py`` is kept deliberately thin — all
network I/O lives here so it can be exercised under pytest without spinning
up Streamlit. Errors are normalised to ``ApiClientError`` so callers do not
have to import ``httpx`` themselves.
"""

from __future__ import annotations

from typing import Any, cast

import httpx


class ApiClientError(RuntimeError):
    """Raised when the API returns a non-success status or is unreachable."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ApiClient:
    """Thin wrapper around the factory-sensor-anomaly REST API."""

    def __init__(self, base_url: str, *, timeout: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def base_url(self) -> str:
        return self._base_url

    # ------------------------------------------------------------------
    # Endpoint wrappers
    # ------------------------------------------------------------------
    def healthz(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._get("/healthz"))

    def list_readings(
        self,
        machine_id: str,
        *,
        sensor_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        params: dict[str, str | int] = {"limit": limit}
        if sensor_name:
            params["sensor_name"] = sensor_name
        return cast(list[dict[str, Any]], self._get(f"/readings/{machine_id}", params=params))

    def list_anomalies(
        self,
        machine_id: str,
        *,
        limit: int = 200,
        only_anomalies: bool = False,
    ) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self._get(
                f"/anomalies/{machine_id}",
                params={"limit": limit, "only_anomalies": str(only_anomalies).lower()},
            ),
        )

    def ingest(self, machine_id: str, readings: list[dict[str, Any]]) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._post("/ingest", json={"machine_id": machine_id, "readings": readings}),
        )

    def infer(
        self,
        machine_id: str,
        sensor_name: str,
        *,
        window_size: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"machine_id": machine_id, "sensor_name": sensor_name}
        if window_size is not None:
            body["window_size"] = window_size
        return cast(dict[str, Any], self._post("/infer", json=body))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, *, json: dict[str, Any]) -> Any:
        return self._request("POST", path, json=json)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = self._client.request(method, path, params=params, json=json)
        except httpx.HTTPError as exc:
            raise ApiClientError(f"{method} {path} failed: {exc}") from exc

        if response.status_code >= 400:
            raise ApiClientError(
                f"{method} {path} -> HTTP {response.status_code}: {response.text}",
                status_code=response.status_code,
            )
        return response.json()
