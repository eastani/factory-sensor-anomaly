"""Tests for ImageApiSettings env-var loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from factory_anomaly.config import ImageApiSettings, get_image_api_settings


def test_defaults_when_no_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "IMAGE_API_MODEL_PATH",
        "IMAGE_API_INPUT_SIZE",
        "IMAGE_API_LOG_LEVEL",
        "IMAGE_API_TORCH_NUM_THREADS",
    ):
        monkeypatch.delenv(var, raising=False)
    s = ImageApiSettings()
    assert s.model_path == Path("models/image_bank.joblib")
    assert s.input_size == 224
    assert s.log_level == "INFO"
    assert s.torch_num_threads == 2


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_API_MODEL_PATH", "/tmp/x.joblib")  # noqa: S108
    monkeypatch.setenv("IMAGE_API_INPUT_SIZE", "128")
    monkeypatch.setenv("IMAGE_API_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("IMAGE_API_TORCH_NUM_THREADS", "4")
    s = ImageApiSettings()
    assert str(s.model_path) == "/tmp/x.joblib"  # noqa: S108
    assert s.input_size == 128
    assert s.log_level == "DEBUG"
    assert s.torch_num_threads == 4


def test_rejects_input_size_below_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_API_INPUT_SIZE", "16")
    with pytest.raises(ValueError):
        ImageApiSettings()


def test_get_image_api_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_API_INPUT_SIZE", "256")
    get_image_api_settings.cache_clear()
    a = get_image_api_settings()
    monkeypatch.setenv("IMAGE_API_INPUT_SIZE", "512")
    b = get_image_api_settings()  # still cached
    assert a is b
    assert a.input_size == 256
    get_image_api_settings.cache_clear()
