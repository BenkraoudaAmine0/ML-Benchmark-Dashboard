"""
Compact, chart-by-chart context for AI interpretation.

Every entry mirrors the numbers actually drawn by ``frontend/charts.py`` so the
model explains the real data instead of guessing from a screenshot. The payload
is intentionally small: rounded numbers, top-N lists, no prediction samples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.data_quality import (
    compare_variants,
    compact_quality,
    primary_label_for,
    primary_metric_for,
)
from core.storage import get_benchmark_history

# Chart key -> where a reader finds it. Used by the AI page and the optional
# "AI read" captions under charts.
CHART_LOCATIONS: Dict[str, str] = {
    "metric_comparison": "Comparison Dashboard · Comparative visualizations",
    "latency": "Comparison Dashboard · Comparative visualizations",
    "radar": "Comparison Dashboard · Multi-metric comparison",
    "category_aggregate": "Comparison Dashboard · Multi-metric comparison",
    "score_vs_time": "Comparison Dashboard · Cost vs. quality",
    "heatmap": "Comparison Dashboard · Benchmark matrix",
    "head_to_head": "Comparison Dashboard · Head-to-head matchup",
    "confusion_matrix": "Model Deep Dive · Confusion matrix",
    "residuals": "Model Deep Dive · Actual vs predicted",
    "feature_importances": "Model Deep Dive · Feature importances",
    "experiment_history": "Experiment History · Benchmark progress",
    "variant_comparison": "Noise & Data Quality · Baseline vs treated",
    "variant_delta": "Noise & Data Quality · Paired delta",
    "noise_profile": "Noise & Data Quality · Noise profile",
    "column_distribution": "Noise & Data Quality · Before / after distribution",
}

METRIC_LABELS = {
    "f1_macro": "F1 (macro)",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "roc_auc": "ROC-AUC",
    "log_loss": "Log loss",
    "r2": "R²",
    "explained_variance": "Explained variance",
    "rmse": "RMSE",
    "mae": "MAE",
    "mse": "MSE",
    "median_ae": "Median absolute error",
    "max_error": "Max error",
    "mape": "MAPE",
}


def _round(value: Any, digits: int = 4) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number):
            return round(number, digits)
    return None


def _metric(row: Dict[str, Any], key: str) -> Optional[float]:
    return _round((row.get("metrics") or {}).get(key))


def _primary(task_type: str) -> str:
    return primary_metric_for(task_type)


def _higher_is_better(metric: str) -> bool:
    return metric not in ("rmse", "mae", "mse", "median_ae", "max_error", "log_loss")


def _best(leaderboard: List[Dict[str, Any]], metric: str, higher: bool = True) -> Optional[Dict[str, Any]]:
    rows = [row for row in leaderboard if _metric(row, metric) is not None]
    if not rows:
        return None
    return max(rows, key=lambda row: _metric(row, metric)) if higher else min(
        rows, key=lambda row: _metric(row, metric)
    )


def _model_summary(row: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    metrics = row.get("metrics") or {}
    if task_type == "classification":
        keys = [
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "roc_auc",
            "log_loss",
            "train_f1_macro",
            "train_accuracy",
        ]
    else:
        keys = [
            "r2",
            "explained_variance",
            "rmse",
            "mae",
            "mse",
            "mape",
            "train_r2",
            "train_rmse",
        ]
    payload = {
        "model": row.get("model_name"),
        "category": row.get("category"),
        "train_time_sec": _round(row.get("train_time_sec"), 4),
        "variant": row.get("variant", "baseline"),
        "overfit_gap": _round(metrics.get("overfit_gap")),
        "is_overfitting": bool(metrics.get("is_overfitting", False)),
    }
    for key in keys:
        payload[key] = _round(metrics.get(key))
    return payload


def _pareto(points: List[Dict[str, Any]]) -> List[str]:
    """Models not dominated on (higher score, lower time)."""
    ordered = sorted(points, key=lambda p: (p["time"], -p["score"]))
    frontier: List[str] = []
    best_score = -float("inf")
    for point in ordered:
        if point["score"] > best_score:
            frontier.append(point["model"])
            best_score = point["score"]
    return frontier


def _ranks(leaderboard: List[Dict[str, Any]], task_type: str) -> Dict[str, Dict[str, int]]:
    if task_type == "classification":
        specs = [
            ("f1_macro", True),
            ("accuracy", True),
            ("balanced_accuracy", True),
            ("precision_macro", True),
            ("recall_macro", True),
            ("roc_auc", True),
            ("log_loss", False),
        ]
    else:
        specs = [
            ("r2", True),
            ("explained_variance", True),
            ("rmse", False),
            ("mae", False),
            ("mse", False),
            ("median_ae", False),
            ("max_error", False),
            ("mape", False),
        ]

    table: Dict[str, Dict[str, int]] = {}
    for key, higher in specs:
        rows = [row for row in leaderboard if _metric(row, key) is not None]
        rows.sort(key=lambda row: _metric(row, key), reverse=higher)
        table[key] = {
            row.get("model_name", row.get("model_id", "")): index + 1
            for index, row in enumerate(rows)
        }
    return table


def _chart_metric_comparison(leaderboard: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
    metric = _primary(task_type)
    values = {
        row.get("model_name"): _metric(row, metric)
        for row in leaderboard
        if _metric(row, metric) is not None
    }
    values = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
    return {
        "title": f"Model comparison · {METRIC_LABELS.get(metric, metric)}",
        "description": (
            "Vertical bars comparing every model on the primary metric; bars are "
            "sorted best to worst."
        ),
        "data": values,
    }


def _chart_latency(leaderboard: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = {
        row.get("model_name"): _round(row.get("train_time_sec"))
        for row in leaderboard
    }
    values = dict(sorted(values.items(), key=lambda item: item[1] or 0))
    return {
        "title": "Training duration · lower is faster",
        "description": "Horizontal bars of wall-clock training time per model.",
        "data": values,
    }


def _chart_radar(leaderboard: List[Dict[str, Any]], task_type: str) -> Optional[Dict[str, Any]]:
    metric = _primary(task_type)
    ranked = sorted(
        [row for row in leaderboard if _metric(row, metric) is not None],
        key=lambda row: _metric(row, metric),
        reverse=True,
    )[:5]
    if not ranked:
        return None
    keys = (
        ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"]
        if task_type == "classification"
        else ["r2", "explained_variance", "mae", "rmse", "mape"]
    )
    return {
        "title": "Multi-metric radar · top models",
        "description": (
            "Each series is a model; each axis a metric normalised to the range "
            "across the shown models (lower-is-better metrics are inverted)."
        ),
        "data": {
            "models": [
                {
                    "model": row.get("model_name"),
                    **{key: _metric(row, key) for key in keys},
                }
                for row in ranked
            ],
            "metrics": [METRIC_LABELS.get(key, key) for key in keys],
        },
    }


def _chart_category(leaderboard: List[Dict[str, Any]], task_type: str) -> Optional[Dict[str, Any]]:
    metric = _primary(task_type)
    groups: Dict[str, List[float]] = {}
    for row in leaderboard:
        value = _metric(row, metric)
        if value is None:
            continue
        groups.setdefault(str(row.get("category") or "unknown"), []).append(value)
    if not groups:
        return None
    return {
        "title": "Classical vs ensemble aggregate",
        "description": (
            "Mean primary score per architecture category with each individual "
            "model plotted as a dot."
        ),
        "data": {
            category: {
                "mean": round(float(np.mean(values)), 4),
                "models": len(values),
                "best": round(float(max(values)), 4),
                "worst": round(float(min(values)), 4),
            }
            for category, values in groups.items()
        },
    }


def _chart_score_vs_time(leaderboard: List[Dict[str, Any]], task_type: str) -> Optional[Dict[str, Any]]:
    metric = _primary(task_type)
    points = []
    for row in leaderboard:
        score = _metric(row, metric)
        time_value = _round(row.get("train_time_sec"))
        if score is None or time_value is None:
            continue
        points.append(
            {
                "model": row.get("model_name"),
                "category": row.get("category"),
                "score": score,
                "time": time_value,
            }
        )
    if not points:
        return None
    frontier = _pareto(points)
    for point in points:
        point["pareto"] = point["model"] in frontier
    return {
        "title": "Cost vs. quality · score against training time",
        "description": (
            "Scatter of primary score against training time; the dotted Pareto "
            "frontier links models nothing else beats on both axes."
        ),
        "data": {"points": points, "frontier": frontier},
    }


def _chart_heatmap(leaderboard: List[Dict[str, Any]], task_type: str) -> Optional[Dict[str, Any]]:
    ranks = _ranks(leaderboard, task_type)
    ranks = {key: value for key, value in ranks.items() if value}
    if not ranks:
        return None
    return {
        "title": "Benchmark matrix · rank per metric",
        "description": (
            "Each cell is the model's rank on that metric; #1 is best. "
            "Lower-is-better metrics are ranked in the correct direction."
        ),
        "data": {"rank_1_is_best": True, "ranks": ranks},
    }


def _chart_head_to_head(
    leaderboard: List[Dict[str, Any]], task_type: str
) -> Optional[Dict[str, Any]]:
    metric = _primary(task_type)
    ranked = sorted(
        [row for row in leaderboard if _metric(row, metric) is not None],
        key=lambda row: _metric(row, metric),
        reverse=True,
    )
    if len(ranked) < 2:
        return None
    first, second = ranked[0], ranked[1]
    keys = (
        ["f1_macro", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "roc_auc", "log_loss"]
        if task_type == "classification"
        else ["r2", "explained_variance", "rmse", "mae", "mse", "mape"]
    )
    deltas = {}
    for key in keys:
        a, b = _metric(first, key), _metric(second, key)
        if a is None or b is None:
            continue
        deltas[key] = {
            "a": a,
            "b": b,
            "delta": round(a - b, 4),
            "higher_is_better": _higher_is_better(key),
        }
    return {
        "title": f"Head-to-head · {first.get('model_name')} vs {second.get('model_name')}",
        "description": (
            "Relative advantage of model A over model B per metric, normalised by "
            "the larger value; bars right mean A wins."
        ),
        "data": {
            "model_a": first.get("model_name"),
            "model_b": second.get("model_name"),
            "deltas": deltas,
            "time_a": _round(first.get("train_time_sec")),
            "time_b": _round(second.get("train_time_sec")),
        },
    }


def _chart_confusion(model: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not model:
        return None
    metrics = model.get("metrics") or {}
    matrix = metrics.get("confusion_matrix")
    if not matrix:
        return None
    report = metrics.get("classification_report") or {}
    classes = [
        key
        for key in report.keys()
        if key not in ("accuracy", "macro avg", "weighted avg", "micro avg")
    ]
    return {
        "title": f"Confusion matrix · {model.get('model_name')}",
        "description": "Rows are actual classes, columns predicted; cells are counts.",
        "data": {"classes": classes, "matrix": matrix},
    }


def _chart_residuals(model: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not model:
        return None
    metrics = model.get("metrics") or {}
    sample = model.get("predictions_sample") or {}
    actual = sample.get("actual") or []
    predicted = sample.get("predicted") or []
    if not actual or not predicted:
        return None
    try:
        residuals = [round(float(a) - float(p), 4) for a, p in zip(actual, predicted)]
    except (TypeError, ValueError):
        # Classification predictions are labels, not residuals.
        return None
    return {
        "title": f"Actual vs predicted · {model.get('model_name')}",
        "description": (
            "Scatter of predicted against actual values on a 50-row test sample, "
            "plus the residual distribution; points on the diagonal are perfect."
        ),
        "data": {
            "sample_size": len(actual),
            "residual_mean": _round(metrics.get("residual_mean")),
            "residual_std": _round(metrics.get("residual_std")),
            "mae": _round(metrics.get("mae")),
            "rmse": _round(metrics.get("rmse")),
            "r2": _round(metrics.get("r2")),
            "first_10_actual": [_round(value) for value in actual[:10]],
            "first_10_predicted": [_round(value) for value in predicted[:10]],
            "first_10_residuals": residuals[:10],
        },
    }


def _chart_importances(model: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not model:
        return None
    importances = model.get("feature_importances") or {}
    if not importances:
        return None
    top = dict(
        sorted(importances.items(), key=lambda item: item[1], reverse=True)[:10]
    )
    return {
        "title": f"Feature importances · {model.get('model_name')}",
        "description": "Top feature contributions, normalised to sum to one.",
        "data": {key: _round(value, 5) for key, value in top.items()},
    }


def _chart_noise_profile(data_quality: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    variants = data_quality.get("variants") or {}
    baseline_cols = (variants.get("baseline") or {}).get("columns") or {}
    treated_cols = (variants.get("treated") or {}).get("columns") or {}
    if not baseline_cols and not treated_cols:
        return None
    columns = list(dict.fromkeys(list(baseline_cols.keys()) + list(treated_cols.keys())))
    rows = {}
    for column in columns[:20]:
        base = baseline_cols.get(column) or {}
        treated = treated_cols.get(column) or {}
        rows[column] = {
            "hf_noise_ratio_before": base.get("hf_noise_ratio"),
            "hf_noise_ratio_after": treated.get("hf_noise_ratio"),
            "roughness_before": base.get("roughness"),
            "roughness_after": treated.get("roughness"),
            "outlier_iqr_pct_before": base.get("outlier_iqr_pct"),
            "outlier_iqr_pct_after": treated.get("outlier_iqr_pct"),
            "missing_pct": base.get("missing_pct"),
        }
    return {
        "title": "Noise profile · high-frequency energy per feature",
        "description": (
            "High-frequency noise ratio (std of the residual after a rolling median "
            "divided by the column std) before and after treatment. Lower means "
            "smoother; hover shows the IQR outlier percentage."
        ),
        "data": rows,
    }


def _chart_distribution(data_quality: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    histograms = (
        (data_quality.get("variants") or {}).get("treated") or {}
    ).get("histograms") or {}
    if not histograms:
        return None
    compact: Dict[str, Any] = {}
    for column, payload in list(histograms.items())[:3]:
        compact[column] = {
            "bin_centers": payload.get("centers", [])[::4],
            "counts_before": payload.get("before", [])[::4],
            "counts_after": (payload.get("after") or [])[::4],
        }
    return {
        "title": "Before / after distribution · treated features",
        "description": (
            "Overlaid histograms of a feature before and after the treatment step; "
            "a narrower or shifted after-distribution shows what smoothing/clipping did."
        ),
        "data": compact,
    }


def build_insight_context(
    experiment: Dict[str, Any],
    leaderboard: List[Dict[str, Any]],
    baseline_leaderboard: Optional[List[Dict[str, Any]]] = None,
    treated_leaderboard: Optional[List[Dict[str, Any]]] = None,
    focus_model: Optional[str] = None,
    include_history: bool = True,
) -> Dict[str, Any]:
    """Assemble the compact payload handed to the AI model."""
    task_type = experiment.get("task_type") or "classification"
    primary = _primary(task_type)
    higher = _higher_is_better(primary)

    best = _best(leaderboard, primary, higher)
    focus = None
    if focus_model:
        focus = next(
            (row for row in leaderboard if row.get("model_name") == focus_model), None
        )
    focus = focus or best

    # Split the stored JSON blobs.
    import json

    def _load(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value or "{}")
        except Exception:
            return {}

    data_quality = _load(experiment.get("data_quality_json"))
    noise_config = _load(experiment.get("noise_config_json"))
    imputer_config = _load(experiment.get("imputer_config_json"))

    charts: Dict[str, Any] = {}
    for key, builder in (
        ("metric_comparison", lambda: _chart_metric_comparison(leaderboard, task_type)),
        ("latency", lambda: _chart_latency(leaderboard)),
        ("radar", lambda: _chart_radar(leaderboard, task_type)),
        ("category_aggregate", lambda: _chart_category(leaderboard, task_type)),
        ("score_vs_time", lambda: _chart_score_vs_time(leaderboard, task_type)),
        ("heatmap", lambda: _chart_heatmap(leaderboard, task_type)),
        ("head_to_head", lambda: _chart_head_to_head(leaderboard, task_type)),
        ("confusion_matrix", lambda: _chart_confusion(focus)),
        ("residuals", lambda: _chart_residuals(focus)),
        ("feature_importances", lambda: _chart_importances(focus)),
        ("noise_profile", lambda: _chart_noise_profile(data_quality)),
        ("column_distribution", lambda: _chart_distribution(data_quality)),
    ):
        try:
            payload = builder()
        except Exception:
            payload = None
        if payload:
            charts[key] = payload

    comparison = None
    if baseline_leaderboard and treated_leaderboard:
        comparison = compare_variants(
            baseline_leaderboard, treated_leaderboard, primary, higher
        )
        charts["variant_comparison"] = {
            "title": "Baseline vs treated · paired scores",
            "description": (
                "Grouped bars of the primary metric for every model, baseline "
                "(noisy/untreated) beside treated (denoised/outlier-handled)."
            ),
            "data": {
                row["model_name"]: {
                    "baseline": row["baseline"],
                    "treated": row["treated"],
                    "delta": row["delta"],
                    "direction": row["direction"],
                }
                for row in comparison["models"]
            },
        }
        charts["variant_delta"] = {
            "title": "Paired delta · treated minus baseline",
            "description": (
                "Per-model change after treatment; positive means the treated "
                "variant improved the primary metric."
            ),
            "data": {
                "primary_metric": comparison["primary_metric"],
                "higher_is_better": comparison["higher_is_better"],
                "mean_delta": comparison["mean_delta"],
                "median_delta": comparison["median_delta"],
                "wins": comparison["wins"],
                "losses": comparison["losses"],
                "unchanged": comparison["unchanged"],
                "rows": comparison["models"],
            },
        }

    history: List[Dict[str, Any]] = []
    if include_history:
        try:
            history = [
                {
                    "id": row["id"],
                    "run_name": row["run_name"],
                    "dataset": row["dataset_name"],
                    "task_type": row["task_type"],
                    "best_score": row["best_score"],
                    "best_model": row["best_model"],
                    "created_at": row["created_at"],
                }
                for row in get_benchmark_history()[-10:]
            ]
        except Exception:
            history = []
        if history:
            charts["experiment_history"] = {
                "title": "Benchmark progress · best score per experiment",
                "description": (
                    "One point per stored experiment, split by task type; hover "
                    "shows the run, dataset and winning model."
                ),
                "data": history,
            }

    payload: Dict[str, Any] = {
        "experiment": {
            "id": experiment.get("id"),
            "run_name": experiment.get("run_name"),
            "dataset": experiment.get("dataset_name"),
            "task_type": task_type,
            "target_column": experiment.get("target_col"),
            "n_train": experiment.get("n_train"),
            "n_test": experiment.get("n_test"),
            "split_id": experiment.get("split_id"),
            "comparison_mode": bool(experiment.get("comparison_mode")),
            "primary_metric": primary,
            "primary_label": primary_label_for(task_type),
            "created_at": experiment.get("created_at"),
            "imputer_config": imputer_config,
            "noise_config": noise_config,
        },
        "leaderboard": [_model_summary(row, task_type) for row in leaderboard],
        "charts": charts,
        "data_quality": compact_quality(data_quality),
        "chart_catalog": {
            key: {"title": value.get("title"), "location": CHART_LOCATIONS.get(key, "")}
            for key, value in charts.items()
        },
    }
    if comparison:
        payload["comparison"] = {
            "primary_metric": comparison["primary_metric"],
            "higher_is_better": comparison["higher_is_better"],
            "n_models": comparison["n_models"],
            "mean_delta": comparison["mean_delta"],
            "median_delta": comparison["median_delta"],
            "wins": comparison["wins"],
            "losses": comparison["losses"],
            "unchanged": comparison["unchanged"],
            "baseline_best": comparison["baseline_best"],
            "treated_best": comparison["treated_best"],
            "best_changed": comparison["best_changed"],
            "models": comparison["models"],
        }
    if history:
        payload["experiment_history"] = history
    return payload
