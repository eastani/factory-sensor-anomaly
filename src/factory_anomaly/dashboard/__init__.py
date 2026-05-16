"""Streamlit dashboard package.

Only the API *client* lives in importable code — the Streamlit view is a
script at ``dashboard/app.py`` because Streamlit's runtime expects a module
file as its entrypoint.
"""

from factory_anomaly.dashboard.client import ApiClient, ApiClientError

__all__ = ["ApiClient", "ApiClientError"]
