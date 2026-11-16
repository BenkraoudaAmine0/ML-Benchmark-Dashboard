"""
Data-quality diagnostics and baseline-vs-treated comparison.

Everything here is descriptive: it runs at training time, is stored in
``experiments.data_quality_json``, and feeds both the dashboard and the AI
context so neither has to reload the original dataset (which may be an upload
that no longer exists).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import TASK_CLASSIFICATION, TASK_REGRESSION


# ------------------------------------------------------------------------------
# Column profiling
# ------------------------------------------------------------------------------
def _safe_float(value: Any, digits: int = 4) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def profile_series(series: pd.Series) -> Dict[str, Any]:
    """Missingness, outlier rates and high-frequency noise for one column."""
    values = pd.to_numeric(series, errors="coerce")
    n = int(len(values))
    missing = int(values.isna().sum())

    info: Dict[str, Any] = {
        "missing": missing,
        "missing_pct": round(100.0 * missing / n, 4) if n else 0.0,
        "unique": int(values.nunique(dropna=True)),
        "mean": _safe_float(values.mean()),
        "std": _safe_float(values.std()),
        "min": _safe_float(values.min()),
        "max": _safe_float(values.max()),
        "skew": _safe_float(values.skew()),
        "kurtosis": _safe_float(values.kurtosis()),
        "outlier_iqr_pct": None,
        "outlier_z_pct": None,
        "roughness": None,
        "hf_noise_ratio": None,
    }

    clean = values.dropna()
    if len(clean) >= 4:
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            info["outlier_iqr_pct"] = round(
                100.0 * float(((clean < low) | (clean > high)).mean()), 4
            )
        std = float(clean.std())
        if std > 0:
            z = (clean - clean.mean()) / std
            info["outlier_z_pct"] = round(100.0 * float((z.abs() > 3.0).mean()), 4)
            diffs = clean.diff().dropna()
            info["roughness"] = _safe_float(float(diffs.std()) / std)
            residual = clean - clean.rolling(window=5, min_periods=1, center=True).median()
            info["hf_noise_ratio"] = _safe_float(float(residual.std()) / std)

    return info


def profile_columns(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Profile every numeric column (or the requested subset)."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return {column: profile_series(df[column]) for column in columns if column in df.columns}


# ------------------------------------------------------------------------------
# Dataset profile
# ------------------------------------------------------------------------------
def profile_dataset(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
) -> Dict[str, Any]:
    """Dataset-level quality summary."""
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    missing_cells = int(df.isna().sum().sum())
    total_cells = int(n_rows * n_cols) or 1
    constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1]
    duplicate_rows = int(df.duplicated().sum())

    profile: Dict[str, Any] = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "missing_cells": missing_cells,
        "missing_pct": round(100.0 * missing_cells / total_cells, 4),
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_cols,
        "target": {},
        "columns": {},
    }

    if target_col in df.columns:
        target = df[target_col]
        if task_type == TASK_CLASSIFICATION:
            counts = target.astype(str).value_counts()
            total = int(counts.sum()) or 1
            majority = float(counts.max()) / total if len(counts) else 0.0
            minority = float(counts.min()) / total if len(counts) else 0.0
            ratio = (majority / minority) if minority > 0 else float("inf")
            if ratio < 1.5:
                severity = "Balanced"
            elif ratio < 3:
                severity = "Mild Imbalance"
            elif ratio < 10:
                severity = "Moderate Imbalance"
            else:
                severity = "Severe Imbalance"
            profile["target"] = {
                "classes": {str(k): int(v) for k, v in counts.items()},
                "n_classes": int(len(counts)),
                "majority_pct": round(100.0 * majority, 4),
                "minority_pct": round(100.0 * minority, 4),
                "imbalance_ratio": _safe_float(ratio, 3) if np.isfinite(ratio) else None,
                "severity": severity,
                "missing_pct": round(100.0 * float(target.isna().mean()), 4),
            }
        elif task_type == TASK_REGRESSION:
            values = pd.to_numeric(target, errors="coerce")
            profile["target"] = {
                "missing_pct": round(100.0 * float(target.isna().mean()), 4),
                "mean": _safe_float(values.mean()),
                "std": _safe_float(values.std()),
                "min": _safe_float(values.min()),
                "max": _safe_float(values.max()),
            }

    return profile


# ------------------------------------------------------------------------------
# Histograms for the before/after distribution chart
# ------------------------------------------------------------------------------
def histogram_summary(
    before: pd.Series,
    after: Optional[pd.Series] = None,
    bins: int = 40,
) -> Optional[Dict[str, Any]]:
    """Shared-bin histogram of a column before and after treatment."""
    before_values = pd.to_numeric(before, errors="coerce").dropna().to_numpy(dtype=float)
    after_values = (
        pd.to_numeric(after, errors="coerce").dropna().to_numpy(dtype=float)
        if after is not None
        else None
    )
    combined = (
        np.concatenate([before_values, after_values])
        if after_values is not None and len(after_values)
        else before_values
    )
    if len(combined) < 2:
        return None

    low, high = float(np.min(combined)), float(np.max(combined))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return None

    edges = np.linspace(low, high, bins + 1)
    before_counts, _ = np.histogram(before_values, bins=edges)
    payload: Dict[str, Any] = {
        "edges": [round(float(e), 6) for e in edges],
        "centers": [round(float((edges[i] + edges[i + 1]) / 2), 6) for i in range(bins)],
        "before": [int(v) for v in before_counts],
        "after": None,
    }
    if after_values is not None and len(after_values):
        after_counts, _ = np.histogram(after_values, bins=edges)
        payload["after"] = [int(v) for v in after_counts]
    return payload


def attach_histograms(
    quality: Dict[str, Any],
    before_frame: pd.DataFrame,
    after_frame: pd.DataFrame,
    columns: List[str],
    max_columns: int = 8,
) -> Dict[str, Any]:
    """Store compact before/after histograms for the treated variant."""
    histograms: Dict[str, Any] = {}
    for column in list(columns)[:max_columns]:
        if column not in before_frame.columns or column not in after_frame.columns:
            continue
        summary = histogram_summary(before_frame[column], after_frame[column])
        if summary:
            histograms[column] = summary
    if histograms:
        quality.setdefault("variants", {}).setdefault("treated", {})[
            "histograms"
        ] = histograms
    return quality


# ------------------------------------------------------------------------------
# Baseline vs treated comparison
# ------------------------------------------------------------------------------
def primary_metric_for(task_type: str) -> str:
    return "f1_macro" if task_type == TASK_CLASSIFICATION else "r2"


def primary_label_for(task_type: str) -> str:
    return "F1-Score (Macro)" if task_type == TASK_CLASSIFICATION else "R² Score"


def _metric(record: Dict[str, Any], key: str) -> Optional[float]:
    value = (record.get("metrics") or {}).get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def compare_variants(
    baseline: List[Dict[str, Any]],
    treated: List[Dict[str, Any]],
    primary_metric: str,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Paired per-model comparison between two variants of the same split."""
    baseline_by_id = {row["model_id"]: row for row in baseline}
    treated_by_id = {row["model_id"]: row for row in treated}
    model_ids = [mid for mid in baseline_by_id if mid in treated_by_id]

    rows: List[Dict[str, Any]] = []
    deltas: List[float] = []
    wins = losses = neutral = 0

    for mid in model_ids:
        b_row, t_row = baseline_by_id[mid], treated_by_id[mid]
        b_val = _metric(b_row, primary_metric)
        t_val = _metric(t_row, primary_metric)
        if b_val is None or t_val is None:
            continue
        delta = t_val - b_val
        if abs(delta) < 1e-9:
            direction = "unchanged"
            neutral += 1
        elif (delta > 0) == bool(higher_is_better):
            direction = "improved"
            wins += 1
        else:
            direction = "degraded"
            losses += 1
        deltas.append(delta)
        rows.append(
            {
                "model_id": mid,
                "model_name": b_row.get("model_name", mid),
                "category": b_row.get("category", ""),
                "baseline": round(b_val, 4),
                "treated": round(t_val, 4),
                "delta": round(delta, 4),
                "pct_change": round(100.0 * delta / abs(b_val), 2) if b_val else None,
                "direction": direction,
                "baseline_time": round(float(b_row.get("train_time_sec") or 0.0), 4),
                "treated_time": round(float(t_row.get("train_time_sec") or 0.0), 4),
            }
        )

    def _best(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        ranked = [
            (rec.get("model_name", ""), _metric(rec, primary_metric))
            for rec in records
            if _metric(rec, primary_metric) is not None
        ]
        if not ranked:
            return {"model": None, "score": None}
        best = max(ranked, key=lambda item: item[1])
        return {"model": best[0], "score": round(float(best[1]), 4)}

    baseline_best = _best(baseline)
    treated_best = _best(treated)

    def _rank_key(record: Dict[str, Any]) -> float:
        value = _metric(record, primary_metric)
        return -1e18 if value is None else value

    baseline_rank = {
        row["model_id"]: index + 1
        for index, row in enumerate(
            sorted(baseline, key=_rank_key, reverse=higher_is_better)
        )
    }
    treated_rank = {
        row["model_id"]: index + 1
        for index, row in enumerate(
            sorted(treated, key=_rank_key, reverse=higher_is_better)
        )
    }
    for row in rows:
        row["rank_baseline"] = baseline_rank.get(row["model_id"])
        row["rank_treated"] = treated_rank.get(row["model_id"])

    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    return {
        "primary_metric": primary_metric,
        "higher_is_better": bool(higher_is_better),
        "models": rows,
        "n_models": len(rows),
        "mean_delta": round(mean_delta, 4),
        "median_delta": round(float(np.median(deltas)), 4) if deltas else 0.0,
        "wins": wins,
        "losses": losses,
        "unchanged": neutral,
        "baseline_best": baseline_best,
        "treated_best": treated_best,
        "best_changed": baseline_best.get("model") != treated_best.get("model"),
    }


def compact_quality(quality: Dict[str, Any], max_columns: int = 25) -> Dict[str, Any]:
    """Trim a stored quality payload for the AI prompt."""
    if not quality:
        return {}
    trimmed = {
        key: value
        for key, value in quality.items()
        if key not in ("columns", "variants")
    }
    columns = quality.get("columns") or {}
    trimmed["columns"] = dict(list(columns.items())[:max_columns])

    variants: Dict[str, Any] = {}
    for name, payload in (quality.get("variants") or {}).items():
        variants[name] = {
            k: v
            for k, v in payload.items()
            if k != "histograms"
        }
        columns = payload.get("columns") or {}
        variants[name]["columns"] = dict(list(columns.items())[:max_columns])
    if variants:
        trimmed["variants"] = variants
    return trimmed
