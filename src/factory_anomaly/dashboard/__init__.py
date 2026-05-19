"""Streamlit dashboard package.

Only the API *client* lives in importable code — the Streamlit view is a
script at ``dashboard/app.py`` because Streamlit's runtime expects a module
file as its entrypoint.
"""

from factory_anomaly.dashboard.client import ApiClient, ApiClientError
from factory_anomaly.dashboard.image_view import (
    DemoImage,
    compose_heatmap_figure,
    downsample_for_upload,
    is_stub_model,
    list_demo_images,
)

__all__ = [
    "ApiClient",
    "ApiClientError",
    "DemoImage",
    "compose_heatmap_figure",
    "downsample_for_upload",
    "is_stub_model",
    "list_demo_images",
]
