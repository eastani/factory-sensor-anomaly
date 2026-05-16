"""Tests for the SKAB loader.

These do not download real SKAB. A tiny in-repo fixture mimics the upstream
schema (semicolon-separated, ``datetime`` index, anomaly column) so the
loader's behaviour is locked in regardless of whether the dataset has been
fetched locally.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factory_anomaly.data.skab import load_skab_csv, load_skab_directory


def _write_fixture(path: Path) -> None:
    path.write_text(
        "datetime;Accelerometer1RMS;Current;Pressure;anomaly;changepoint\n"
        "2020-01-01 00:00:00;0.10;1.2;3.1;0;0\n"
        "2020-01-01 00:00:01;0.11;1.3;3.2;0;0\n"
        "2020-01-01 00:00:02;5.50;9.8;0.1;1;1\n"
        "2020-01-01 00:00:03;5.55;9.7;0.1;1;0\n"
    )


def test_load_skab_csv_parses_datetime_and_normalises_anomaly(tmp_path: Path) -> None:
    fixture = tmp_path / "valve1_0.csv"
    _write_fixture(fixture)

    df = load_skab_csv(fixture)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.shape == (4, 5)
    assert df["anomaly"].dtype == bool
    assert df["anomaly"].sum() == 2
    assert "Accelerometer1RMS" in df.columns


def test_load_skab_directory_walks_subdirs(tmp_path: Path) -> None:
    _write_fixture(tmp_path / "a.csv")
    sub = tmp_path / "other"
    sub.mkdir()
    _write_fixture(sub / "b.csv")

    frames = load_skab_directory(tmp_path)
    assert set(frames) == {"a", "other/b"}


def test_load_skab_directory_errors_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no CSV files"):
        load_skab_directory(tmp_path)


def test_load_skab_directory_errors_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_skab_directory(tmp_path / "does-not-exist")
