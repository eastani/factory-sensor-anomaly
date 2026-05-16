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

import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from factory_anomaly.dashboard import ApiClient, ApiClientError
from factory_anomaly.dashboard.demo_data import generate_and_send

DEFAULT_MACHINE_ID = "pump-001"
DEFAULT_SENSOR_NAME = "sensor_00"


# ---------------------------------------------------------------------------
# Cached resources / fetchers
# ---------------------------------------------------------------------------


@st.cache_resource
def get_client() -> ApiClient:
    base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    return ApiClient(base_url=base_url)


@st.cache_data(ttl=5)
def fetch_health() -> dict[str, Any] | None:
    try:
        return get_client().healthz()
    except ApiClientError:
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


if __name__ == "__main__":
    main()
