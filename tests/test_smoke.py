"""Smoke tests — verify the package imports and the toolchain works."""

from __future__ import annotations

import factory_anomaly


def test_version_is_set() -> None:
    assert factory_anomaly.__version__
    assert isinstance(factory_anomaly.__version__, str)


def test_version_follows_semver_shape() -> None:
    parts = factory_anomaly.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
