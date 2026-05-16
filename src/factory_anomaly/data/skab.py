"""Loader for the Skoltech Anomaly Benchmark (SKAB) dataset.

Data is *not* committed to the repo (GPL-3.0 + size). Use ``scripts/evaluate_skab.py``
to download it on demand from ``waico/SKAB`` and run an evaluation. This module
just turns one directory of SKAB CSV files into clean, typed dataframes.

SKAB CSV schema (per the upstream README):

- ``datetime`` — ISO timestamp index
- one column per sensor channel (Accelerometer1RMS, Accelerometer2RMS,
  Current, Pressure, Temperature, Thermocouple, Voltage, Volume Flow RateRMS)
- ``anomaly`` — 1 where the engineer marked the sample anomalous, else 0
- ``changepoint`` — 1 at the *start* of an anomalous interval; used by the
  benchmark scoring function but not needed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

SKAB_SENSOR_COLUMNS: Final[tuple[str, ...]] = (
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
)


def load_skab_csv(path: Path) -> pd.DataFrame:
    """Load one SKAB CSV with a tz-naive datetime index and bool anomaly column."""
    df = pd.read_csv(path, sep=";", index_col="datetime", parse_dates=["datetime"])
    df.index = pd.to_datetime(df.index)

    # ``anomaly`` lands as float in some SKAB files; normalise to bool.
    if "anomaly" in df.columns:
        df["anomaly"] = df["anomaly"].astype(int).astype(bool)
    return df


def load_skab_directory(directory: Path) -> dict[str, pd.DataFrame]:
    """Load every ``.csv`` under ``directory`` into a dict keyed by file stem.

    Subdirectories (e.g. ``other/`` for non-anomalous baseline files) are
    walked recursively so the caller can still partition them by name.
    """
    if not directory.exists():
        raise FileNotFoundError(f"SKAB directory not found: {directory}")

    out: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(directory.rglob("*.csv")):
        key = csv_path.relative_to(directory).with_suffix("").as_posix()
        out[key] = load_skab_csv(csv_path)

    if not out:
        raise FileNotFoundError(f"no CSV files found under {directory}")
    return out
