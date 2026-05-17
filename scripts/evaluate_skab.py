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
from factory_anomaly.ml.features import make_rolling_features

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


REPORT_TEMPLATE = """# SKAB evaluation — sensor `{sensor}`

Auto-generated by ``scripts/evaluate_skab.py``. Train each detector on the
normal-only baseline files (``data/anomaly-free/``); evaluate on the labelled
runs (``other/``, ``valve1/``, ``valve2/``).

| Setting | Value |
|---|---|
| Sensor | `{sensor}` |
| Window size | {window} |
| STL period | {period} |
| Anomaly share | {baseline_anomaly_share:.1%} |

## Results

| Detector | Eval windows | Precision | Recall | F1 | ROC AUC |
|----------|-------------:|----------:|-------:|---:|--------:|
| **Baseline IF** (Phase 1.6) | {baseline_n_eval_windows:,} | {baseline_precision:.3f} | {baseline_recall:.3f} | **{baseline_f1:.3f}** | **{baseline_roc_auc}** |
| **STL + IF** (Phase 1.7) | {stl_n_eval_windows:,} | {stl_precision:.3f} | {stl_recall:.3f} | **{stl_f1:.3f}** | **{stl_roc_auc}** |
| Δ | — | {delta_precision:+.3f} | {delta_recall:+.3f} | **{delta_f1:+.3f}** | **{delta_roc_auc}** |

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

## What this means for Phase 1.8

- **Multivariate features** are now the most promising next move — concatenate
  rolling features across multiple sensors and re-evaluate. Single-channel
  is the binding constraint here, not periodicity.
- STL stays in the codebase (it is the right tool on signals with stronger
  periodic confounds — the synthetic sine-wave preview in the README is one
  such example, where STL would help). The honest evaluation has just
  established that SKAB isn't that kind of dataset.

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

    if args.no_stl:
        stl = EvalResult(
            name="stl-skipped",
            n_eval_windows=0,
            anomaly_share=0.0,
            precision=float("nan"),
            recall=float("nan"),
            f1=float("nan"),
            roc_auc=None,
        )
    else:
        print(f"Evaluating STL+IF (period={args.period}) ...")
        stl = evaluate_stl(frames, sensor=args.sensor, window=args.window, period=args.period)
        print(
            f"  precision={stl.precision:.3f} recall={stl.recall:.3f} "
            f"f1={stl.f1:.3f} auc={_fmt_optional(stl.roc_auc)}"
        )

    delta_roc_auc = _fmt_delta(stl.roc_auc, baseline.roc_auc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        REPORT_TEMPLATE.format(
            sensor=args.sensor,
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
            delta_precision=stl.precision - baseline.precision,
            delta_recall=stl.recall - baseline.recall,
            delta_f1=stl.f1 - baseline.f1,
            delta_roc_auc=delta_roc_auc,
        )
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
