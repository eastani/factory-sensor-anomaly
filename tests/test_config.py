"""Unit tests for configuration loading — no DB required."""

from __future__ import annotations

import pytest

from factory_anomaly.config import DatabaseSettings


def test_defaults_compose_a_valid_psycopg_url() -> None:
    settings = DatabaseSettings()
    url = settings.url
    assert url.startswith("postgresql+psycopg://")
    assert settings.db in url
    assert str(settings.port) in url


def test_overrides_via_constructor() -> None:
    settings = DatabaseSettings(
        user="alice",
        password="s3cret",
        db="metrics",
        host="db.internal",
        port=6543,
    )
    url = settings.url
    assert "alice" in url
    assert "s3cret" in url
    assert "metrics" in url
    assert "db.internal" in url
    assert "6543" in url


def test_env_var_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_USER", "from_env")
    monkeypatch.setenv("POSTGRES_DB", "envdb")
    # Bypass any .env file by pointing pydantic at a nonexistent path.
    settings = DatabaseSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.user == "from_env"
    assert settings.db == "envdb"
