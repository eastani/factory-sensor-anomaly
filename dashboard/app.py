"""Streamlit entry point.

Run locally:

    API_BASE_URL=http://localhost:8000 \
        uv run streamlit run dashboard/app.py

Design notes (see ADR-0003):

- The dashboard is *only* a view onto the API. It owns no business logic.
- Auto-refresh uses ``st.fragment(run_every=...)`` rather than a naive
  ``st.autorefresh`` loop — only the chart fragment re-runs, not the whole
  page. This avoids the flicker + N-API-calls-per-second problem.
- ``/healthz`` is cached so we don't ping it on every fragment tick.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from factory_anomaly.dashboard import (
    ApiClient,
    ApiClientError,
    compose_heatmap_figure,
    downsample_for_upload,
    is_stub_model,
    list_demo_images,
)
from factory_anomaly.dashboard.demo_data import generate_and_send
from factory_anomaly.image.client import ImageApiClient, ImageApiClientError

DEFAULT_MACHINE_ID = "pump-001"
DEFAULT_SENSOR_NAME = "sensor_00"


# ---------------------------------------------------------------------------
# Cached resources / fetchers
# ---------------------------------------------------------------------------


@st.cache_resource
def get_client() -> ApiClient:
    base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    return ApiClient(base_url=base_url)


@st.cache_resource
def get_image_client() -> ImageApiClient:
    base_url = os.environ.get("IMAGE_API_BASE_URL", "http://localhost:8001")
    return ImageApiClient(base_url=base_url)


@st.cache_data(ttl=5)
def fetch_health() -> dict[str, Any] | None:
    try:
        return get_client().healthz()
    except ApiClientError:
        return None


@st.cache_data(ttl=10)
def fetch_image_health() -> dict[str, Any] | None:
    try:
        return get_image_client().healthz()
    except ImageApiClientError:
        return None


def fetch_readings(machine_id: str, sensor_name: str, limit: int) -> pd.DataFrame:
    rows = get_client().list_readings(machine_id, sensor_name=sensor_name, limit=limit)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "value"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def fetch_anomalies(machine_id: str, limit: int) -> pd.DataFrame:
    rows = get_client().list_anomalies(machine_id, limit=limit)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "score", "is_anomaly"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def render_header() -> None:
    st.title("Factory Sensor Anomaly")
    st.caption(
        "Real-time anomaly detection on synthetic factory-pump sensor streams. "
        "Click **Generate demo data** to populate a fresh batch."
    )

    health = fetch_health()
    cols = st.columns(3)
    if health is None:
        cols[0].error("API unreachable")
    else:
        emoji = "🟢" if health["status"] == "ok" else "🟡"
        cols[0].metric("API status", f"{emoji} {health['status']}")
        cols[1].metric("Database", health["db"])
        cols[2].metric("Model", health.get("model_version") or "—")


def render_sidebar() -> tuple[str, str, int, int]:
    st.sidebar.header("Configuration")
    machine_id = st.sidebar.text_input("Machine ID", value=DEFAULT_MACHINE_ID)
    sensor_name = st.sidebar.text_input("Sensor name", value=DEFAULT_SENSOR_NAME)
    history_limit = st.sidebar.slider("Reading history limit", 50, 1000, 200, step=50)
    refresh_seconds = st.sidebar.slider("Refresh interval (s)", 2, 30, 5)

    st.sidebar.markdown("---")
    if st.sidebar.button("Generate demo data", use_container_width=True):
        try:
            with st.spinner("Calling /ingest and /infer..."):
                summary = generate_and_send(get_client(), machine_id)
            st.sidebar.success(
                f"Ingested {summary['ingested']} readings; "
                f"score={summary['score']:.3f}, anomaly={summary['is_anomaly']}"
            )
            fetch_health.clear()
        except ApiClientError as exc:
            st.sidebar.error(f"API error: {exc}")

    return machine_id, sensor_name, history_limit, refresh_seconds


def build_chart(readings: pd.DataFrame, anomalies: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not readings.empty:
        fig.add_trace(
            go.Scatter(
                x=readings["timestamp"],
                y=readings["value"],
                mode="lines",
                name="sensor",
                line={"width": 1.5},
            )
        )

    if not anomalies.empty:
        anomalous = anomalies[anomalies["is_anomaly"]]
        if not anomalous.empty and not readings.empty:
            joined = pd.merge_asof(
                anomalous.sort_values("timestamp"),
                readings.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=5),
            ).dropna(subset=["value"])
            if not joined.empty:
                fig.add_trace(
                    go.Scatter(
                        x=joined["timestamp"],
                        y=joined["value"],
                        mode="markers",
                        name="anomaly",
                        marker={"color": "red", "size": 9, "symbol": "x"},
                    )
                )

    fig.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title="timestamp",
        yaxis_title="value",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def build_score_chart(anomalies: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["score"],
                mode="lines+markers",
                name="anomaly score",
                line={"width": 1.5},
            )
        )
    fig.update_layout(
        height=240,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title="timestamp",
        yaxis_title="score (higher = more anomalous)",
    )
    return fig


def render_image_panel() -> None:
    """Image anomaly section — file uploader + bundled demo gallery.

    On-demand only (no auto-refresh). The image API itself does not
    persist predictions (Phase 3.2 decision), so "latest" has no
    server-side meaning; the dashboard scores whatever the operator
    submits and caches the result in ``st.session_state``.
    """
    st.markdown("---")
    st.subheader("Image anomaly (PatchCore)")

    health = fetch_image_health()
    if health is None:
        st.warning(
            "Image API unreachable. Check that `image-api` is running "
            "(`docker compose ps image-api`) and that `IMAGE_API_BASE_URL` "
            "points at it."
        )
        return
    if not health.get("model_loaded"):
        st.warning(
            "Image API is up but no memory bank is loaded. Either bake one "
            "into the image (`make image-train-stub`) or mount a real bank "
            "over `/app/models/image_bank.joblib`."
        )
        return

    model_version = health.get("model_version") or ""
    if is_stub_model(model_version):
        st.info(
            f"Loaded model is a **stub bank** (`{model_version}`) trained on "
            "the bundled demo plates. Heatmaps are meaningful for the demo "
            "images below; for production-quality numbers, swap in a bank "
            "from `make image-evaluate`."
        )

    cols = st.columns([1, 2])
    with cols[0]:
        source = st.radio(
            "Image source",
            options=["Bundled demo image", "Upload your own"],
            index=0,
            key="image_source_radio",
        )

        image_bytes: bytes | None = None
        source_label: str = ""

        if source == "Bundled demo image":
            demos = list_demo_images()
            if not demos:
                st.warning(
                    "No demo images found under `docs/assets/demo-images/test/`. "
                    "Run `uv run python scripts/generate_demo_images.py`."
                )
            else:
                labels = [d.label for d in demos]
                choice = st.selectbox("Pick one", labels, key="image_demo_picker")
                selected = next(d for d in demos if d.label == choice)
                image_bytes = selected.read_bytes()
                source_label = selected.path.name
        else:
            uploaded = st.file_uploader(
                "Upload PNG or JPEG (< 5 MB)",
                type=["png", "jpg", "jpeg"],
                key="image_uploader",
            )
            if uploaded is not None:
                raw = uploaded.getvalue()
                if len(raw) > 5 * 1024 * 1024:
                    st.error("File exceeds 5 MB; resize before uploading.")
                else:
                    image_bytes = downsample_for_upload(raw)
                    source_label = uploaded.name

        run_clicked = st.button(
            "Run inference",
            type="primary",
            use_container_width=True,
            disabled=image_bytes is None,
        )

    with cols[1]:
        if not run_clicked and "image_result" not in st.session_state:
            st.caption(
                "Pick a demo image or upload one, then click **Run inference**. "
                "The 3-panel result shows the input, the per-patch anomaly map, "
                "and an overlay highlighting where the model thinks the defect is."
            )
            return

        if run_clicked and image_bytes is not None:
            cache_key = hashlib.sha256(image_bytes).hexdigest()
            cached = st.session_state.get("image_result")
            if cached is None or cached["cache_key"] != cache_key:
                try:
                    with st.spinner("Scoring..."):
                        result = get_image_client().predict(
                            image_bytes, filename=source_label or "upload.png"
                        )
                except ImageApiClientError as exc:
                    st.error(f"Image API error: {exc}")
                    return
                st.session_state["image_result"] = {
                    "cache_key": cache_key,
                    "image_bytes": image_bytes,
                    "source_label": source_label,
                    "score": float(result["score"]),
                    "model_version": result["model_version"],
                    "anomaly_map": np.asarray(result["anomaly_map"], dtype=np.float32),
                    "elapsed_ms": float(result.get("elapsed_ms", 0.0)),
                }

        cached = st.session_state.get("image_result")
        if cached is None:
            return

        fig = compose_heatmap_figure(
            cached["image_bytes"],
            cached["anomaly_map"],
            cached["score"],
            model_version=cached["model_version"],
        )
        st.pyplot(fig, use_container_width=True)
        st.caption(
            f"Source: {cached['source_label']}  ·  "
            f"server-side latency: {cached['elapsed_ms']:.0f} ms"
        )


def main() -> None:
    st.set_page_config(page_title="Factory Sensor Anomaly", layout="wide")
    render_header()
    machine_id, sensor_name, history_limit, refresh_seconds = render_sidebar()

    @st.fragment(run_every=refresh_seconds)
    def live_panel() -> None:
        try:
            readings = fetch_readings(machine_id, sensor_name, history_limit)
            anomalies = fetch_anomalies(machine_id, history_limit)
        except ApiClientError as exc:
            st.error(f"API error: {exc}")
            return

        if readings.empty:
            st.info(
                "No readings yet for this machine_id/sensor. Click "
                "**Generate demo data** in the sidebar."
            )
            return

        st.subheader("Sensor stream")
        st.plotly_chart(build_chart(readings, anomalies), use_container_width=True)

        st.subheader("Anomaly score history")
        st.plotly_chart(build_score_chart(anomalies), use_container_width=True)

    live_panel()
    render_image_panel()


if __name__ == "__main__":
    main()
