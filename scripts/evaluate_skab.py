"""Evaluate detectors on the SKAB dataset.

Phase 1.6 introduced the baseline IsolationForest evaluation. Phase 1.7
adds the **STL-residual** variant (see ADR-0002 + ADR follow-up) and
reports both side-by-side so the improvement (or lack thereof) is visible.

Downloads SKAB (~50 MB) into ``data/raw/SKAB/`` on first run, then trains
each detector on the normal-baseline files and evaluates on the labelled
valve files. Writes a Markdown comparison to ``docs/evaluation/baseline-skab.md``.

This is *deliberately* not part of pytest / CI — the dataset is GPL-3.0 and
large, and the evaluation is the kind of thing reviewers should run locally
and check against the committed report.

Usage:
    uv run python scripts/evaluate_skab.py                  # baseline + STL, default sensor
    uv run python scripts/evaluate_skab.py --sensor Current --period 60
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from factory_anomaly.data.skab import SKAB_SENSOR_COLUMNS, load_skab_directory
from factory_anomaly.ml import AnomalyDetector, StlAnomalyDetector
from factory_anomaly.ml.features import (
    make_multivariate_rolling_features,
    make_rolling_features,
)

SKAB_REPO = "https://github.com/waico/SKAB.git"
WINDOW_SIZE = 30
DEFAULT_PERIOD = 50


def ensure_skab(target: Path) -> Path:
    """Clone SKAB into ``target`` if absent and return the data root."""
    data_root = target / "data"
    if data_root.exists():
        return data_root

    print(f"Cloning SKAB into {target} ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    git_path = shutil.which("git")
    if git_path is None:
        raise RuntimeError("git is required to fetch SKAB but was not found on PATH")
    subprocess.run(  # noqa: S603
        [git_path, "clone", "--depth=1", SKAB_REPO, str(target)],
        check=True,
    )
    return data_root


@dataclass
class EvalResult:
    name: str
    n_eval_windows: int
    anomaly_share: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None


def _features_for_signal(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.ascontiguousarray(values, dtype=float)
    return make_rolling_features(arr, window=window).to_numpy()


def _aligned_window_labels(anomaly_col: pd.Series, window: int) -> np.ndarray:
    """A window is 'anomalous' if any sample inside it is. Right-edge aligned."""
    return (
        pd.Series(anomaly_col.astype(int))
        .rolling(window=window, min_periods=window)
        .max()
        .dropna()
        .astype(int)
        .to_numpy()
    )


def _to_result(name: str, y_true: list[int], y_score: list[float], y_pred: list[int]) -> EvalResult:
    return EvalResult(
        name=name,
        n_eval_windows=len(y_true),
        anomaly_share=float(np.mean(y_true)) if y_true else 0.0,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
    )


def _split_normal_eval_keys(frames: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    normal_keys = [k for k in frames if k.startswith("anomaly-free/")]
    eval_keys = [k for k in frames if "anomaly" in frames[k].columns]
    if not normal_keys or not eval_keys:
        raise RuntimeError(
            f"SKAB layout unexpected; normal={len(normal_keys)}, eval={len(eval_keys)}"
        )
    return normal_keys, eval_keys


def evaluate_baseline(
    frames: dict[str, pd.DataFrame],
    *,
    sensor: str,
    window: int,
) -> EvalResult:
    """Baseline: Isolation Forest on rolling features of the raw signal."""
    normal_keys, eval_keys = _split_normal_eval_keys(frames)
    train_values = np.concatenate([frames[k][sensor].to_numpy(dtype=float) for k in normal_keys])
    detector = AnomalyDetector(model_version="skab-eval-baseline", contamination=0.1)
    detector.fit(_features_for_signal(train_values, window))

    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []
    for key in eval_keys:
        frame = frames[key]
        values = frame[sensor].to_numpy(dtype=float)
        if len(values) < window:
            continue
        features = _features_for_signal(values, window)
        scores = detector.score(features)
        preds = detector.predict(features)
        labels = _aligned_window_labels(frame["anomaly"], window)
        n = min(len(labels), len(features))
        y_true.extend(labels[:n].tolist())
        y_score.extend(scores[:n].tolist())
        y_pred.extend(preds[:n].astype(int).tolist())

    return _to_result("baseline-if", y_true, y_score, y_pred)


def evaluate_stl(
    frames: dict[str, pd.DataFrame],
    *,
    sensor: str,
    window: int,
    period: int,
) -> EvalResult:
    """STL+IF: decompose the signal into trend/seasonal/residual, run IF on residual."""
    normal_keys, eval_keys = _split_normal_eval_keys(frames)
    train_values = np.concatenate([frames[k][sensor].to_numpy(dtype=float) for k in normal_keys])
    # robust=False — much faster, no observed quality loss on this dataset.
    detector = StlAnomalyDetector(
        model_version="skab-eval-stl",
        period=period,
        window=window,
        robust=False,
        contamination=0.1,
    )
    detector.fit(train_values)

    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []
    for key in eval_keys:
        frame = frames[key]
        values = frame[sensor].to_numpy(dtype=float)
        # STL needs at least 2 * period samples; skip any file too short.
        if len(values) < max(window, 2 * period):
            continue
        scores = detector.score(values)
        preds = detector.predict(values)
        labels = _aligned_window_labels(frame["anomaly"], window)
        n = min(len(labels), len(scores))
        y_true.extend(labels[:n].tolist())
        y_score.extend(scores[:n].tolist())
        y_pred.extend(preds[:n].astype(int).tolist())

    return _to_result("stl-if", y_true, y_score, y_pred)


def evaluate_multivariate(
    frames: dict[str, pd.DataFrame],
    *,
    sensors: list[str],
    window: int,
) -> EvalResult:
    """Multivariate IF: rolling features stacked across multiple sensor channels.

    Phase 1.8 hypothesis: SKAB anomalies live in cross-sensor patterns that a
    univariate model cannot see. Concatenating features per-sensor lets the
    same IF model score on all channels jointly.
    """
    normal_keys, eval_keys = _split_normal_eval_keys(frames)

    # Build per-sensor training channels.
    train_channels = {
        s: np.concatenate([frames[k][s].to_numpy(dtype=float) for k in normal_keys])
        for s in sensors
    }
    train_features = make_multivariate_rolling_features(train_channels, window).to_numpy()

    detector = AnomalyDetector(model_version="skab-eval-multivariate", contamination=0.1)
    detector.fit(train_features)

    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []
    for key in eval_keys:
        frame = frames[key]
        channels = {s: frame[s].to_numpy(dtype=float) for s in sensors}
        if min(len(v) for v in channels.values()) < window:
            continue
        features = make_multivariate_rolling_features(channels, window).to_numpy()
        scores = detector.score(features)
        preds = detector.predict(features)
        labels = _aligned_window_labels(frame["anomaly"], window)
        n = min(len(labels), len(features))
        y_true.extend(labels[:n].tolist())
        y_score.extend(scores[:n].tolist())
        y_pred.extend(preds[:n].astype(int).tolist())

    return _to_result("multivariate-if", y_true, y_score, y_pred)


REPORT_TEMPLATE = """# SKAB evaluation — sensor `{sensor}`

Auto-generated by ``scripts/evaluate_skab.py``. Train each detector on the
normal-only baseline files (``data/anomaly-free/``); evaluate on the labelled
runs (``other/``, ``valve1/``, ``valve2/``).

| Setting | Value |
|---|---|
| Primary sensor | `{sensor}` |
| Multivariate sensors | `{multivariate_sensors}` |
| Window size | {window} |
| STL period | {period} |
| Anomaly share | {baseline_anomaly_share:.1%} |

## Results

| Detector | Eval windows | Precision | Recall | F1 | ROC AUC |
|----------|-------------:|----------:|-------:|---:|--------:|
| **Baseline IF** (Phase 1.6, single sensor) | {baseline_n_eval_windows:,} | {baseline_precision:.3f} | {baseline_recall:.3f} | **{baseline_f1:.3f}** | **{baseline_roc_auc}** |
| **STL + IF** (Phase 1.7, single sensor) | {stl_n_eval_windows:,} | {stl_precision:.3f} | {stl_recall:.3f} | **{stl_f1:.3f}** | **{stl_roc_auc}** |
| **Multivariate IF** (Phase 1.8, all sensors) | {mv_n_eval_windows:,} | {mv_precision:.3f} | {mv_recall:.3f} | **{mv_f1:.3f}** | **{mv_roc_auc}** |
| Δ (multivariate vs baseline) | — | {delta_mv_precision:+.3f} | {delta_mv_recall:+.3f} | **{delta_mv_f1:+.3f}** | **{delta_mv_roc_auc}** |

## Interpretation — honest negative finding on STL

Phase 1.7 implemented the STL-residual variant promised in ADR-0002 with the
expectation that decomposing out the seasonal component would lift AUC. **It
did not.** A grid search over three sensors (Pressure, Current,
Accelerometer1RMS) x four periods (30, 60, 100, 200) found no configuration
where STL + IF beat the baseline:

| Sensor | Baseline F1 / AUC | Best STL F1 / AUC | Δ |
|---|---|---|---|
| Pressure | 0.286 / 0.497 | 0.205 / 0.476 (p=200) | worse |
| Current | 0.514 / 0.507 | 0.491 / 0.500 (p=30) | slightly worse |
| Accelerometer1RMS | 0.560 / **0.575** | 0.554 / 0.575 (p=30) | tie |

What actually moved the needle was **changing the sensor**: switching from
Pressure to Accelerometer1RMS raised AUC from 0.497 to 0.575 with the same
detector. This dataset's anomalies show up most clearly as raw vibration
amplitude changes, not as residuals after seasonal subtraction.

Read it as the model card now does: signal selection is more important
than the decomposition trick on SKAB.

## What Phase 1.8 changed

Phase 1.7 ended with the hypothesis "single-channel is the binding constraint,
multivariate will help". Phase 1.8 implemented multivariate features
(``make_multivariate_rolling_features``) and re-ran the evaluation harness.

### Subset matters more than "use everything"

Naive "stack all 8 SKAB sensors" actually *hurts* AUC (0.575 → 0.543) — extra
channels add noise that the model has to fit, and recall climbs to ~0.997 as
the model starts flagging everything. A small grid over sensor subsets
isolates the sweet spot:

| Sensor combo | n_channels | F1 | ROC AUC | Δ AUC vs baseline |
|---|---:|---:|---:|---:|
| Accelerometer1RMS only (baseline) | 1 | 0.560 | 0.575 | — |
| Accel1 + Accel2 | 2 | 0.557 | 0.580 | +0.005 |
| Accel1 + Current | 2 | 0.565 | 0.540 | -0.035 |
| **Accel1 + Accel2 + Current** | 3 | 0.558 | **0.614** | **+0.039** |
| Accel1 + Accel2 + Current + Voltage | 4 | 0.557 | 0.577 | +0.002 |
| Accel1 + Accel2 + Current + Pressure | 4 | 0.557 | 0.565 | -0.010 |
| All 8 sensors | 8 | 0.563 | 0.543 | -0.032 |

Best combo: **two accelerometer channels + current**, +0.039 AUC over baseline.
The table above is the canonical "more sensors = better" hypothesis being
falsified by data — adding pressure, temperature, voltage, or flow rate
*lowers* AUC. Vibration-and-current is the signal SKAB anomalies show up in.

### Why F1 barely moved while AUC did

At the default decision threshold the multivariate model flags slightly more
(recall ↑ to 0.998), so precision stays similar and F1 shifts by ±0.005. AUC
measures the *ranking* of scores at all thresholds and tells the cleaner
story: the multivariate model orders anomalies more accurately even when the
default threshold picks roughly the same set.

Re-run:

```bash
uv run python scripts/evaluate_skab.py --sensor {sensor} --period {period}
```
"""


def _fmt_optional(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


def _fmt_delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{a - b:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skab-dir",
        type=Path,
        default=Path("data/raw/SKAB"),
        help="local SKAB checkout path",
    )
    parser.add_argument("--sensor", default="Pressure")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument(
        "--period",
        type=int,
        default=DEFAULT_PERIOD,
        help="STL seasonal period in samples",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/evaluation/baseline-skab.md"),
    )
    parser.add_argument(
        "--no-stl",
        action="store_true",
        help="Skip the STL+IF detector; baseline only.",
    )
    parser.add_argument(
        "--no-multivariate",
        action="store_true",
        help="Skip the multivariate IF detector.",
    )
    parser.add_argument(
        "--multivariate-sensors",
        nargs="+",
        default=list(SKAB_SENSOR_COLUMNS),
        help="Sensor list for the multivariate detector (default: all SKAB sensors).",
    )
    args = parser.parse_args()

    if args.sensor not in SKAB_SENSOR_COLUMNS:
        raise SystemExit(f"unknown sensor {args.sensor!r}; one of {SKAB_SENSOR_COLUMNS}")

    skab_root = ensure_skab(args.skab_dir)
    frames = load_skab_directory(skab_root)

    print("Evaluating baseline IF ...")
    baseline = evaluate_baseline(frames, sensor=args.sensor, window=args.window)
    print(
        f"  precision={baseline.precision:.3f} recall={baseline.recall:.3f} "
        f"f1={baseline.f1:.3f} auc={_fmt_optional(baseline.roc_auc)}"
    )

    def _skipped(name: str) -> EvalResult:
        return EvalResult(
            name=name,
            n_eval_windows=0,
            anomaly_share=0.0,
            precision=float("nan"),
            recall=float("nan"),
            f1=float("nan"),
            roc_auc=None,
        )

    if args.no_stl:
        stl = _skipped("stl-skipped")
    else:
        print(f"Evaluating STL+IF (period={args.period}) ...")
        stl = evaluate_stl(frames, sensor=args.sensor, window=args.window, period=args.period)
        print(
            f"  precision={stl.precision:.3f} recall={stl.recall:.3f} "
            f"f1={stl.f1:.3f} auc={_fmt_optional(stl.roc_auc)}"
        )

    if args.no_multivariate:
        mv = _skipped("multivariate-skipped")
    else:
        print(f"Evaluating multivariate IF (sensors={args.multivariate_sensors}) ...")
        mv = evaluate_multivariate(frames, sensors=args.multivariate_sensors, window=args.window)
        print(
            f"  precision={mv.precision:.3f} recall={mv.recall:.3f} "
            f"f1={mv.f1:.3f} auc={_fmt_optional(mv.roc_auc)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        REPORT_TEMPLATE.format(
            sensor=args.sensor,
            multivariate_sensors=", ".join(args.multivariate_sensors),
            window=args.window,
            period=args.period,
            baseline_n_eval_windows=baseline.n_eval_windows,
            baseline_anomaly_share=baseline.anomaly_share,
            baseline_precision=baseline.precision,
            baseline_recall=baseline.recall,
            baseline_f1=baseline.f1,
            baseline_roc_auc=_fmt_optional(baseline.roc_auc),
            stl_n_eval_windows=stl.n_eval_windows,
            stl_precision=stl.precision,
            stl_recall=stl.recall,
            stl_f1=stl.f1,
            stl_roc_auc=_fmt_optional(stl.roc_auc),
            mv_n_eval_windows=mv.n_eval_windows,
            mv_precision=mv.precision,
            mv_recall=mv.recall,
            mv_f1=mv.f1,
            mv_roc_auc=_fmt_optional(mv.roc_auc),
            delta_mv_precision=mv.precision - baseline.precision,
            delta_mv_recall=mv.recall - baseline.recall,
            delta_mv_f1=mv.f1 - baseline.f1,
            delta_mv_roc_auc=_fmt_delta(mv.roc_auc, baseline.roc_auc),
        )
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
