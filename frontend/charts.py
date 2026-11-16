"""
Plotly charts for the ML2 Model Benchmark Studio
=================================================
Every figure follows one visual system — graphite surfaces, cyan / blue
accents, mono uppercase captions and transparent plotting areas so the app's
frame cards show through. Dark and light palettes are chosen at runtime via
``set_theme()``; the app calls it once per Streamlit script run.
"""

from __future__ import annotations

import contextvars
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"
DISPLAY = "'Sora', 'Inter', sans-serif"

# Interactive from the first paint: scroll to zoom, drag to pan, double-click
# to reset, and PNG export at 2x.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "displayModeBar": True,
    "doubleClick": "reset+autosize",
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "filename": "benchmark-chart", "scale": 2},
}

_THEME = contextvars.ContextVar("benchmark_chart_theme", default="dark")

CHART_THEMES: Dict[str, Dict[str, Any]] = {
    "dark": {
        "text": "#F4F6F8",
        "body": "#C3CAD5",
        "muted": "#7C8798",
        "grid": "rgba(255, 255, 255, 0.055)",
        "axis": "rgba(255, 255, 255, 0.14)",
        "surface": "#101216",
        "surface_2": "#14171C",
        "surface_3": "#191D24",
        "line": "rgba(255, 255, 255, 0.075)",
        "cyan": "#22D3EE",
        "pink": "#F472B6",
        "violet": "#A78BFA",
        "green": "#86EFAC",
        "amber": "#FBBF24",
        "orange": "#FB923C",
        "rose": "#FB7185",
        "blue": "#60A5FA",
        "bar_fill": "rgba(34, 211, 238, 0.62)",
        "bar_edge": "rgba(34, 211, 238, 0.95)",
        "radar_fill": 0.20,
        "heat_scale": [
            [0.0, "#191D24"],
            [0.45, "#7C3AED"],
            [1.0, "#22D3EE"],
        ],
    },
    "light": {
        "text": "#0B1220",
        "body": "#42526B",
        "muted": "#64748B",
        "grid": "rgba(15, 23, 42, 0.07)",
        "axis": "#CBD5E1",
        "surface": "#FFFFFF",
        "surface_2": "#F8FAFC",
        "surface_3": "#F1F5F9",
        "line": "#E2E8F0",
        "cyan": "#0891B2",
        "pink": "#DB2777",
        "violet": "#7C3AED",
        "green": "#059669",
        "amber": "#D97706",
        "orange": "#EA580C",
        "rose": "#E11D48",
        "blue": "#2563EB",
        "bar_fill": "rgba(8, 145, 178, 0.55)",
        "bar_edge": "rgba(8, 145, 178, 0.95)",
        "radar_fill": 0.16,
        "heat_scale": [
            [0.0, "#F1F5F9"],
            [0.45, "#7C3AED"],
            [1.0, "#0891B2"],
        ],
    },
}

_CLASSICAL = "blue"
_ENSEMBLE = "orange"
_ACCENT = "amber"

# Metrics the charts offer, in display order. ``higher_is_better`` is used by
# ranking and by the relative-delta chart; unknown keys default to higher-better.
METRIC_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "classification": [
        {"key": "f1_macro", "label": "F1 (Macro)", "higher_is_better": True},
        {"key": "accuracy", "label": "Accuracy", "higher_is_better": True},
        {"key": "balanced_accuracy", "label": "Balanced Acc", "higher_is_better": True},
        {"key": "precision_macro", "label": "Precision", "higher_is_better": True},
        {"key": "recall_macro", "label": "Recall", "higher_is_better": True},
        {"key": "roc_auc", "label": "ROC-AUC", "higher_is_better": True},
        {"key": "log_loss", "label": "Log Loss", "higher_is_better": False},
    ],
    "regression": [
        {"key": "r2", "label": "R²", "higher_is_better": True},
        {"key": "explained_variance", "label": "Expl. Var", "higher_is_better": True},
        {"key": "rmse", "label": "RMSE", "higher_is_better": False},
        {"key": "mae", "label": "MAE", "higher_is_better": False},
        {"key": "mse", "label": "MSE", "higher_is_better": False},
        {"key": "median_ae", "label": "MedAE", "higher_is_better": False},
        {"key": "max_error", "label": "Max Error", "higher_is_better": False},
        {"key": "mape", "label": "MAPE", "higher_is_better": False},
    ],
}


def metric_specs(task_type: str) -> List[Dict[str, Any]]:
    """Metric definitions for a task type, falling back to classification."""
    return METRIC_SPECS.get(str(task_type).lower(), METRIC_SPECS["classification"])


def model_palette(model_names: List[str]) -> Dict[str, str]:
    """Stable model -> hue map: 8 categorical tokens, then muted ink (never cycled).

    The order alternates warm and cool families and keeps the close token pairs
    (cyan/blue, amber/orange, pink/rose) apart, so neighbouring radar series
    stay distinguishable in both themes.
    """
    t = tokens()
    palette = [t["blue"], t["orange"], t["green"], t["violet"],
               t["rose"], t["amber"], t["cyan"], t["pink"]]
    return {
        name: (palette[index] if index < len(palette) else t["muted"])
        for index, name in enumerate(model_names)
    }


def _is_number(value: Any) -> bool:
    """True for int/float metric values, but not for booleans or None."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def set_theme(mode: str) -> None:
    """Set the chart theme for the current Streamlit script thread."""
    _THEME.set("light" if str(mode).lower() == "light" else "dark")


def theme() -> str:
    return _THEME.get()


def tokens() -> Dict[str, Any]:
    return CHART_THEMES[theme()]


def _rgba(color: str, alpha: float) -> str:
    """#RRGGBB -> rgba(); leaves existing colour strings untouched."""
    value = str(color).strip()
    if value.startswith("#") and len(value) == 7:
        try:
            red, green, blue = (int(value[i:i + 2], 16) for i in (1, 3, 5))
            return f"rgba({red},{green},{blue},{alpha})"
        except ValueError:
            return value
    return value


# ------------------------------------------------------------------------------
# Shared styling helpers
# ------------------------------------------------------------------------------
def _polish(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 380,
    hovermode: str = "x unified",
    showlegend: bool = True,
) -> go.Figure:
    """Apply the shared layout: transparent canvas, mono caption, themed axes."""
    t = tokens()
    fig.update_layout(
        template=None,
        height=height,
        dragmode="pan",
        margin=dict(l=14, r=14, t=86 if title else 52, b=26),
        font=dict(family=FONT, size=12, color=t["body"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode=hovermode,
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=t["muted"], family=MONO),
        ),
        hoverlabel=dict(
            bgcolor=t["surface_2"],
            bordercolor=t["line"],
            font=dict(color=t["text"], family=FONT, size=12),
        ),
        modebar=dict(
            orientation="h",
            bgcolor="rgba(0,0,0,0)",
            color=t["muted"],
            activecolor=t["cyan"],
        ),
    )
    if title:
        fig.update_layout(
            title=dict(
                text=str(title).upper(), x=0, xanchor="left",
                y=1.0, yref="container", yanchor="top",
                font=dict(size=10.5, color=t["muted"], family=MONO),
            )
        )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor=t["axis"], linewidth=1,
        tickfont=dict(size=10, color=t["muted"], family=MONO),
        title_font=dict(size=10, color=t["muted"], family=MONO),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=t["grid"], gridwidth=1,
        zeroline=False, showline=False,
        tickfont=dict(size=10, color=t["muted"], family=MONO),
        title_font=dict(size=10, color=t["muted"], family=MONO),
        automargin=True,
    )
    return fig


def _glow(fig: go.Figure, x: Any, y: Any, color: str, width: int = 11, opacity: float = 0.16) -> None:
    """Wide, faint duplicate of a line — the reference's neon halo."""
    if theme() == "light":
        # A neon halo reads as a smear on white paper: light mode goes flat.
        return
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=color, width=width, shape="spline", smoothing=1.0),
            opacity=opacity,
        )
    )


def _empty_figure(message: str, height: int = 320) -> go.Figure:
    """Placeholder card when a chart has no data."""
    t = tokens()
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False,
        font=dict(family=MONO, size=11, color=t["muted"]),
        x=0.5, y=0.5, xref="paper", yref="paper",
    )
    fig.update_layout(
        template=None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=14, r=14, t=26, b=14),
    )
    return fig


# ------------------------------------------------------------------------------
# Public chart builders
# ------------------------------------------------------------------------------
def plot_metric_comparison(
    leaderboard: List[Dict[str, Any]],
    metric_key: str,
    metric_label: Optional[str] = None,
    higher_is_better: bool = True,
) -> go.Figure:
    """Vertical bars comparing one metric across all models."""
    t = tokens()
    rows = []
    for item in leaderboard:
        value = item["metrics"].get(metric_key)
        if value is None:
            continue
        rows.append((float(value), item["model_name"], item.get("category", "classical")))

    if not rows:
        return _empty_figure(f"No values recorded for {metric_key.upper().replace('_', ' ')}")

    rows.sort(key=lambda row: row[0], reverse=higher_is_better)
    names = [row[1] for row in rows]
    values = [row[0] for row in rows]
    colors = [t[_ENSEMBLE] if row[2] == "ensemble" else t[_CLASSICAL] for row in rows]

    label = metric_label or metric_key.upper().replace("_", " ")
    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker=dict(
                color=[_rgba(color, 0.55) for color in colors],
                line=dict(color=colors, width=1.2),
            ),
            text=[f"{value:.4f}" for value in values],
            textposition="outside",
            textfont=dict(family=MONO, size=10, color=t["body"]),
            cliponaxis=False,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.4f}}<extra></extra>",
        )
    )
    fig.update_xaxes(tickangle=-28)

    # Keep the outside labels inside the plot box.
    span = (max(values) - min(values)) or abs(max(values)) or 1.0
    pad = span * 0.16
    fig.update_yaxes(range=[min(values) - pad, max(values) + pad] if min(values) >= 0 else None)

    return _polish(fig, f"Model comparison · {label}", height=400, hovermode="x unified", showlegend=False)


def plot_latency_comparison(leaderboard: List[Dict[str, Any]]) -> go.Figure:
    """Horizontal bars of training time, fastest first."""
    t = tokens()
    if not leaderboard:
        return _empty_figure("No training runs recorded")

    rows = sorted(
        ((float(item["train_time_sec"]), item["model_name"], item.get("category", "classical"))
         for item in leaderboard),
        key=lambda row: row[0],
    )
    names = [row[1] for row in rows]
    times = [row[0] for row in rows]
    colors = [t[_ENSEMBLE] if row[2] == "ensemble" else t[_CLASSICAL] for row in rows]

    fig = go.Figure(
        go.Bar(
            y=names,
            x=times,
            orientation="h",
            marker=dict(
                color=[_rgba(color, 0.55) for color in colors],
                line=dict(color=colors, width=1.2),
            ),
            text=[f"{value:.4f}s" for value in times],
            textposition="outside",
            textfont=dict(family=MONO, size=10, color=t["body"]),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Training time: %{x:.4f}s<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig = _polish(fig, "Training duration · lower is faster", height=400, hovermode="y unified", showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], zeroline=False, showline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_radar_metrics(
    leaderboard: List[Dict[str, Any]],
    task_type: str,
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Radar chart comparing models across several metrics.

    ``colors`` maps model name -> hex token so a model keeps its hue when the
    app filters the list; without it hues are assigned once per call.
    """
    t = tokens()
    if not leaderboard:
        return _empty_figure("No model results to compare", height=420)

    if task_type == "classification":
        metric_keys = ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"]
        labels = ["Accuracy", "Balanced Acc", "F1 (Macro)", "Precision", "Recall"]
        radial_range = [0, 1]
    else:
        metric_keys = ["r2", "explained_variance"]
        labels = ["R² Score", "Explained Variance"]
        radial_range = None

    # Drop axes no model has a value for: a None would otherwise be plotted as 0.
    axes = [
        (key, label)
        for key, label in zip(metric_keys, labels)
        if any(item["metrics"].get(key) is not None for item in leaderboard)
    ]
    if not axes:
        return _empty_figure("No comparable metrics recorded", height=420)
    metric_keys = [key for key, _ in axes]
    labels = [label for _, label in axes]

    if colors is None:
        colors = model_palette([item["model_name"] for item in leaderboard])

    fill_alpha = float(t.get("radar_fill", 0.16))

    fig = go.Figure()
    for item in leaderboard:
        values = [item["metrics"].get(key) if item["metrics"].get(key) is not None else 0.0
                  for key in metric_keys]
        values.append(values[0])
        color = colors.get(item["model_name"], t["blue"])
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor=_rgba(color, fill_alpha),
                name=item["model_name"],
                mode="lines+markers",
                line=dict(color=color, width=2.2),
                marker=dict(
                    color=color,
                    size=5,
                    line=dict(color=t["surface"], width=1.5),
                ),
                hovertemplate="<b>%{theta}</b><br>%{r:.4f}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=radial_range,
                gridcolor=t["axis"],
                linecolor=t["axis"],
                tickfont=dict(size=9, color=t["muted"], family=MONO),
                angle=90,
            ),
            angularaxis=dict(
                gridcolor=t["axis"],
                linecolor=t["axis"],
                tickfont=dict(size=10, color=t["body"], family=MONO),
            ),
        ),
    )
    return _polish(fig, "Multi-metric radar comparison", height=440, hovermode="closest", showlegend=True)


def plot_confusion_matrix(cm: List[List[int]], class_names: Optional[List[str]] = None) -> go.Figure:
    """Row-normalised confusion-matrix heatmap with count + share labels."""
    t = tokens()
    matrix = np.asarray(cm)
    n_classes = matrix.shape[0]
    labels = class_names or [f"Class {i}" for i in range(n_classes)]

    row_totals = np.maximum(1, matrix.sum(axis=1, keepdims=True))
    normalised = matrix.astype(float) / row_totals

    annotations = []
    for i in range(n_classes):
        for j in range(n_classes):
            share = normalised[i, j]
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{matrix[i, j]}</b><br>{share:.1%}",
                    showarrow=False,
                    font=dict(
                        family=MONO,
                        size=11,
                        color=t["text"] if share > 0.55 else t["body"],
                    ),
                )
            )

    fig = go.Figure(
        go.Heatmap(
            z=normalised,
            x=labels,
            y=labels,
            colorscale=t["heat_scale"],
            showscale=True,
            zmin=0,
            zmax=1,
            colorbar=dict(
                thickness=10,
                outlinewidth=0,
                tickfont=dict(family=MONO, size=9, color=t["muted"]),
                tickformat=".0%",
            ),
            hovertemplate="<b>Actual:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Share:</b> %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(annotations=annotations)
    fig.update_xaxes(title="Predicted label")
    fig.update_yaxes(title="Actual label", autorange="reversed")

    fig = _polish(
        fig,
        "Confusion matrix · row share",
        height=max(380, 46 * n_classes + 140),
        hovermode="closest",
        showlegend=False,
    )
    fig.update_yaxes(showgrid=False)
    return fig


def plot_residuals(actual: List[float], predicted: List[float]) -> go.Figure:
    """Actual vs predicted scatter with the perfect-fit diagonal."""
    t = tokens()
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if act.size == 0 or pred.size == 0:
        return _empty_figure("Prediction sample not available", height=400)

    low = float(min(act.min(), pred.min()))
    high = float(max(act.max(), pred.max()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=act,
            y=pred,
            mode="markers",
            name="Predictions",
            marker=dict(
                color=_rgba(t["cyan"], 0.72),
                size=8,
                line=dict(color=t["surface"], width=0.8),
            ),
            hovertemplate="<b>Actual:</b> %{x:.4f}<br><b>Predicted:</b> %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Perfect fit (y = x)",
            line=dict(color=t[_ACCENT], width=2, dash="dash"),
            hovertemplate="Perfect fit<br>%{x:.4f}<extra></extra>",
        )
    )

    return _polish(fig, "Actual vs predicted", height=420, hovermode="closest", showlegend=True)


def plot_feature_importances(importances: Dict[str, float], top_n: int = 15) -> go.Figure:
    """Horizontal bars of the strongest feature importances."""
    t = tokens()
    if not importances:
        return _empty_figure("Feature importances are not available for this model")

    ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:top_n]
    features = [item[0] for item in ranked][::-1]
    scores = [float(item[1]) for item in ranked][::-1]

    fig = go.Figure(
        go.Bar(
            y=features,
            x=scores,
            orientation="h",
            marker=dict(
                color=_rgba(t["cyan"], 0.55),
                line=dict(color=t["cyan"], width=1.2),
            ),
            text=[f"{score:.4f}" for score in scores],
            textposition="outside",
            textfont=dict(family=MONO, size=10, color=t["body"]),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig = _polish(
        fig,
        f"Top {min(top_n, len(features))} feature importances",
        height=max(350, len(features) * 26 + 90),
        hovermode="y unified",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], zeroline=False, showline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_score_vs_time(
    leaderboard: List[Dict[str, Any]],
    metric_key: str,
    metric_label: Optional[str] = None,
    higher_is_better: bool = True,
) -> go.Figure:
    """Score against training time, with the non-dominated (Pareto) frontier."""
    t = tokens()
    label = metric_label or metric_key.upper().replace("_", " ")

    rows: List[Dict[str, Any]] = []
    for item in leaderboard:
        value = item["metrics"].get(metric_key)
        elapsed = item.get("train_time_sec")
        if not _is_number(value) or not _is_number(elapsed):
            continue
        value, elapsed = float(value), float(elapsed)
        if elapsed <= 0:
            continue
        rows.append(
            {
                "name": item["model_name"],
                "category": item.get("category", "classical"),
                "value": value,
                "time": elapsed,
            }
        )

    if not rows:
        return _empty_figure(f"No {label} / training-time pairs recorded", height=420)

    # A point is Pareto-efficient when nothing is both faster and better.
    sign = 1.0 if higher_is_better else -1.0
    frontier: List[Dict[str, Any]] = []
    best = float("-inf")
    for row in sorted(rows, key=lambda r: r["time"]):
        if row["value"] * sign > best:
            best = row["value"] * sign
            frontier.append(row)

    fig = go.Figure()
    for category, token in (("classical", _CLASSICAL), ("ensemble", _ENSEMBLE)):
        members = [row for row in rows if row["category"] == category]
        if not members:
            continue
        color = t[token]
        fig.add_trace(
            go.Scatter(
                x=[row["time"] for row in members],
                y=[row["value"] for row in members],
                mode="markers+text",
                name=category.capitalize(),
                text=[row["name"] for row in members],
                textposition="top center",
                textfont=dict(family=MONO, size=10, color=t["body"]),
                marker=dict(
                    color=_rgba(color, 0.72),
                    size=12,
                    line=dict(color=t["surface"], width=2),
                ),
                customdata=[[row["category"].capitalize()] for row in members],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{label}: %{{y:.4f}}<br>"
                    "Training time: %{x:.4f}s<extra>%{customdata[0]}</extra>"
                ),
            )
        )

    if len(frontier) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[row["time"] for row in frontier],
                y=[row["value"] for row in frontier],
                mode="lines",
                name="Pareto frontier",
                line=dict(color=t[_ACCENT], width=1.6, dash="dot"),
                hoverinfo="skip",
            )
        )

    times = [row["time"] for row in rows]
    use_log = min(times) > 0 and (max(times) / min(times)) > 25
    fig.update_xaxes(
        type="log" if use_log else "linear",
        title_text="Training time · seconds" + (" (log)" if use_log else ""),
    )

    scores = [row["value"] for row in rows]
    low, high = min(scores), max(scores)
    span = (high - low) or abs(high) or 1.0
    fig.update_yaxes(
        title_text=label,
        range=[low - span * 0.12, high + span * 0.18],
    )

    return _polish(
        fig,
        "Score vs. training cost",
        height=460,
        hovermode="closest",
        showlegend=True,
    )


def plot_metric_heatmap(
    leaderboard: List[Dict[str, Any]],
    task_type: str = "classification",
) -> go.Figure:
    """Model × metric matrix, each cell showing the model's rank on that metric."""
    t = tokens()
    if not leaderboard:
        return _empty_figure("No model results to rank", height=380)

    # Keep only metrics that at least one model recorded.
    columns: List[tuple] = []
    for spec in metric_specs(task_type):
        values = [item["metrics"].get(spec["key"]) for item in leaderboard]
        if any(_is_number(value) for value in values):
            columns.append((spec, values))

    if len(columns) < 2:
        return _empty_figure("Not enough metrics recorded to build the matrix", height=380)

    n_models = len(leaderboard)
    rank_matrix = np.full((n_models, len(columns)), np.nan)
    raw_matrix = np.full((n_models, len(columns)), np.nan)

    for column, (spec, values) in enumerate(columns):
        present = [
            (index, float(value))
            for index, value in enumerate(values)
            if _is_number(value)
        ]
        if not present:
            continue
        ordered = sorted(
            present,
            key=lambda pair: pair[1],
            reverse=bool(spec["higher_is_better"]),
        )
        # Competition ranking: equal values share the best rank.
        current_rank = 1
        previous: Optional[float] = None
        for position, (index, value) in enumerate(ordered):
            if previous is None or not np.isclose(value, previous):
                current_rank = position + 1
                previous = value
            rank_matrix[index, column] = current_rank
            raw_matrix[index, column] = value

    names = [item["model_name"] for item in leaderboard]
    row_scores = []
    for row in rank_matrix:
        valid = row[~np.isnan(row)]
        row_scores.append(float(valid.mean()) if valid.size else float("inf"))
    order = np.argsort(np.asarray(row_scores))
    rank_matrix = rank_matrix[order]
    raw_matrix = raw_matrix[order]
    names = [names[index] for index in order]

    x_labels = [spec["label"] for spec, _ in columns]
    # Reversed ramp (stops still ascending): rank 1 lands on the bright end.
    colorscale = sorted(
        ([round(1.0 - stop, 4), color] for stop, color in t["heat_scale"]),
        key=lambda pair: pair[0],
    )

    annotations = []
    for row, name in enumerate(names):
        for column, label in enumerate(x_labels):
            rank = rank_matrix[row, column]
            strong = not np.isnan(rank) and rank <= max(2.0, n_models * 0.25)
            annotations.append(
                dict(
                    x=label,
                    y=name,
                    text="—" if np.isnan(rank) else f"#{int(rank)}",
                    showarrow=False,
                    font=dict(
                        family=MONO,
                        size=10,
                        color=t["surface"] if strong else t["body"],
                    ),
                )
            )

    fig = go.Figure(
        go.Heatmap(
            z=rank_matrix,
            x=x_labels,
            y=names,
            customdata=raw_matrix,
            colorscale=colorscale,
            zmin=1,
            zmax=max(2, n_models),
            showscale=True,
            hoverongaps=False,
            colorbar=dict(
                thickness=10,
                outlinewidth=0,
                tickvals=[1, max(2, n_models)],
                ticktext=["best", "worst"],
                tickfont=dict(family=MONO, size=9, color=t["muted"]),
            ),
            hovertemplate=(
                "<b>%{y}</b> · %{x}<br>"
                "Rank: %{z:.0f}<br>"
                "Value: %{customdata:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(annotations=annotations)

    fig = _polish(
        fig,
        "Benchmark matrix · rank per metric",
        height=max(380, 30 * n_models + 150),
        hovermode="closest",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def plot_category_aggregate(
    leaderboard: List[Dict[str, Any]],
    metric_key: str,
    metric_label: Optional[str] = None,
    higher_is_better: bool = True,
) -> go.Figure:
    """Mean score per model category, with every model shown as a dot."""
    t = tokens()
    label = metric_label or metric_key.upper().replace("_", " ")

    groups = []
    for category, token in (("classical", _CLASSICAL), ("ensemble", _ENSEMBLE)):
        members = [
            item
            for item in leaderboard
            if item.get("category", "classical") == category
            and _is_number(item["metrics"].get(metric_key))
        ]
        if not members:
            continue
        values = [float(item["metrics"][metric_key]) for item in members]
        times = [float(item.get("train_time_sec") or 0.0) for item in members]
        groups.append(
            {
                "category": category,
                "token": token,
                "names": [item["model_name"] for item in members],
                "values": values,
                "mean": float(np.mean(values)),
                "mean_time": float(np.mean(times)),
            }
        )

    if not groups:
        return _empty_figure(f"No {label} values recorded", height=380)

    positions = list(range(len(groups)))
    fig = go.Figure(
        go.Bar(
            x=positions,
            y=[group["mean"] for group in groups],
            width=0.46,
            name="Mean score",
            marker=dict(
                color=[_rgba(t[group["token"]], 0.28) for group in groups],
                line=dict(
                    color=[t[group["token"]] for group in groups],
                    width=1.4,
                ),
            ),
            text=[f"{group['mean']:.4f}" for group in groups],
            textposition="outside",
            textfont=dict(family=MONO, size=11, color=t["body"]),
            cliponaxis=False,
            customdata=[
                [group["category"].capitalize(), group["mean_time"], len(group["values"])]
                for group in groups
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"Mean {label}: %{{y:.4f}}<br>"
                "Mean training time: %{customdata[1]:.4f}s<br>"
                "Models: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    for offset, group in enumerate(groups):
        count = len(group["values"])
        spread = np.linspace(-0.17, 0.17, count) if count > 1 else [0.0]
        fig.add_trace(
            go.Scatter(
                x=[positions[offset] + step for step in spread],
                y=group["values"],
                mode="markers",
                name="Individual models" if offset == 0 else None,
                showlegend=offset == 0,
                marker=dict(
                    color=t[group["token"]],
                    size=9,
                    line=dict(color=t["surface"], width=1.5),
                ),
                text=group["names"],
                hovertemplate=(
                    "<b>%{text}</b><br>" + f"{label}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    all_values = [value for group in groups for value in group["values"]]
    all_values += [group["mean"] for group in groups]
    low, high = min(all_values), max(all_values)
    span = (high - low) or abs(high) or 1.0
    bottom = low - span * 0.16
    fig.update_yaxes(range=[bottom if low < 0 else max(0.0, bottom), high + span * 0.22])
    fig.update_xaxes(
        tickmode="array",
        tickvals=positions,
        ticktext=[group["category"].capitalize() for group in groups],
        range=[-0.6, len(groups) - 0.4],
    )

    return _polish(
        fig,
        f"Classical vs ensemble · mean {label}",
        height=400,
        hovermode="closest",
        showlegend=True,
    )


def plot_head_to_head(
    model_a: Dict[str, Any],
    model_b: Dict[str, Any],
    task_type: str = "classification",
) -> go.Figure:
    """Relative advantage of model A over model B, one diverging bar per metric."""
    t = tokens()
    name_a = model_a["model_name"]
    name_b = model_b["model_name"]
    metrics_a = model_a.get("metrics", {})
    metrics_b = model_b.get("metrics", {})

    # Curated, direction-aware list: headline metrics, then train-side and
    # overfitting diagnostics. Residual statistics are excluded because they
    # have no "better" direction.
    entries = [
        (spec["key"], spec["label"], bool(spec["higher_is_better"]))
        for spec in metric_specs(task_type)
    ]
    if task_type == "classification":
        entries += [
            ("train_f1_macro", "Train F1", True),
            ("train_accuracy", "Train Accuracy", True),
        ]
    else:
        entries += [
            ("train_r2", "Train R²", True),
            ("train_rmse", "Train RMSE", False),
        ]
    entries.append(("overfit_gap", "Overfit Gap", False))

    rows: List[Dict[str, Any]] = []
    for key, label, higher_is_better in entries:
        value_a = metrics_a.get(key)
        value_b = metrics_b.get(key)
        if not _is_number(value_a) or not _is_number(value_b):
            continue
        # Difference divided by the larger magnitude: bounded to ±100%, so one
        # extreme metric cannot squash every other bar.
        scale = max(abs(float(value_a)), abs(float(value_b)))
        if scale < 1e-9:
            continue
        advantage = (
            (float(value_a) - float(value_b)) / scale
            if higher_is_better
            else (float(value_b) - float(value_a)) / scale
        )
        rows.append(
            {
                "label": label.upper(),
                "advantage": advantage,
                "a": float(value_a),
                "b": float(value_b),
            }
        )

    time_a = model_a.get("train_time_sec")
    time_b = model_b.get("train_time_sec")
    if _is_number(time_a) and _is_number(time_b):
        time_scale = max(abs(float(time_a)), abs(float(time_b)))
        if time_scale > 0:
            rows.append(
                {
                    "label": "TRAINING TIME",
                    "advantage": (float(time_b) - float(time_a)) / time_scale,
                    "a": float(time_a),
                    "b": float(time_b),
                }
            )

    if not rows:
        return _empty_figure("No comparable metrics between the two models", height=340)

    rows.sort(key=lambda row: row["advantage"], reverse=True)
    colors = [t["green"] if row["advantage"] >= 0 else t["rose"] for row in rows]

    fig = go.Figure(
        go.Bar(
            x=[row["advantage"] for row in rows],
            y=[row["label"] for row in rows],
            orientation="h",
            marker=dict(
                color=[_rgba(color, 0.55) for color in colors],
                line=dict(color=colors, width=1.2),
            ),
            text=[f"{row['advantage']:+.1%}" for row in rows],
            textposition="outside",
            textfont=dict(family=MONO, size=10, color=t["body"]),
            cliponaxis=False,
            customdata=[[row["a"], row["b"]] for row in rows],
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{name_a}: %{{customdata[0]:.4g}}<br>"
                f"{name_b}: %{{customdata[1]:.4g}}<br>"
                "Advantage A: %{x:+.1%}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=t["axis"])

    values = [row["advantage"] for row in rows]
    low, high = min(values), max(values)
    span = (high - low) or 0.1
    fig.update_xaxes(
        range=[low - span * 0.2, high + span * 0.2],
        tickformat="+.0%",
        title_text="← B better · relative advantage · A better →",
    )
    fig.update_yaxes(showgrid=False)

    fig = _polish(
        fig,
        f"{name_a} vs {name_b} · relative advantage",
        height=max(320, 34 * len(rows) + 130),
        hovermode="y unified",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], zeroline=False, showline=False)
    return fig


def plot_experiment_trend(history: List[Dict[str, Any]]) -> go.Figure:
    """Best primary score per stored experiment, one line per task type."""
    t = tokens()
    rows = [row for row in history if _is_number(row.get("best_score"))]
    if not rows:
        return _empty_figure("No benchmark history recorded yet", height=320)

    fig = go.Figure()
    for task, token, label in (
        ("classification", _ENSEMBLE, "Classification"),
        ("regression", "violet", "Regression"),
    ):
        members = [row for row in rows if row.get("task_type") == task]
        if not members:
            continue
        color = t[token]
        fig.add_trace(
            go.Scatter(
                x=[row["id"] for row in members],
                y=[float(row["best_score"]) for row in members],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(
                    color=color,
                    size=9,
                    line=dict(color=t["surface"], width=2),
                ),
                customdata=[
                    [
                        row.get("run_name", ""),
                        row.get("dataset_name", ""),
                        row.get("best_model", ""),
                        row.get("created_at", ""),
                    ]
                    for row in members
                ],
                hovertemplate=(
                    "<b>#%{x}</b> · %{customdata[0]}<br>"
                    "Dataset: %{customdata[1]}<br>"
                    "Best model: %{customdata[2]}<br>"
                    "Best score: %{y:.4f}<br>"
                    "%{customdata[3]}<extra>" + label + "</extra>"
                ),
            )
        )

    tick_ids = [row["id"] for row in rows][-12:]
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_ids,
        ticktext=[f"#{value}" for value in tick_ids],
        title_text="Experiment",
    )
    fig.update_yaxes(title_text="Best primary score")

    return _polish(
        fig,
        "Benchmark progress · best score per experiment",
        height=340,
        hovermode="x unified",
        showlegend=True,
    )


# ------------------------------------------------------------------------------
# Noise treatment / data-quality charts
# ------------------------------------------------------------------------------
def _metric_map(rows: List[Dict[str, Any]], metric_key: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for row in rows:
        value = (row.get("metrics") or {}).get(metric_key)
        if _is_number(value):
            values[row.get("model_name", row.get("model_id", ""))] = float(value)
    return values


def plot_variant_comparison(
    baseline: List[Dict[str, Any]],
    treated: List[Dict[str, Any]],
    metric_key: str,
    metric_label: Optional[str] = None,
    higher_is_better: bool = True,
    baseline_label: str = "Noisy / untreated",
    treated_label: str = "Treated",
) -> go.Figure:
    """Grouped bars comparing the same models before and after treatment."""
    t = tokens()
    base_values = _metric_map(baseline, metric_key)
    treated_values = _metric_map(treated, metric_key)

    names = list(dict.fromkeys(list(base_values.keys()) + list(treated_values.keys())))
    if not names:
        return _empty_figure(
            f"No {metric_key.upper().replace('_', ' ')} values across variants"
        )

    ordered = sorted(
        names,
        key=lambda name: base_values.get(name, treated_values.get(name, 0.0)),
        reverse=higher_is_better,
    )
    label = metric_label or metric_key.upper().replace("_", " ")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ordered,
            y=[base_values.get(name) for name in ordered],
            name=baseline_label,
            marker=dict(
                color=_rgba(t["blue"], 0.45),
                line=dict(color=t["blue"], width=1.1),
            ),
            text=[_fmt(base_values.get(name), 4) for name in ordered],
            textposition="outside",
            textfont=dict(family=MONO, size=9, color=t["body"]),
            cliponaxis=False,
            hovertemplate=f"<b>%{{x}}</b><br>{baseline_label} {label}: %{{y:.4f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=ordered,
            y=[treated_values.get(name) for name in ordered],
            name=treated_label,
            marker=dict(
                color=_rgba(t["green"], 0.5),
                line=dict(color=t["green"], width=1.1),
            ),
            text=[_fmt(treated_values.get(name), 4) for name in ordered],
            textposition="outside",
            textfont=dict(family=MONO, size=9, color=t["body"]),
            cliponaxis=False,
            hovertemplate=f"<b>%{{x}}</b><br>{treated_label} {label}: %{{y:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(tickangle=-28)
    return _polish(
        fig,
        f"Variant comparison · {label} before and after treatment",
        height=420,
        hovermode="x unified",
        showlegend=True,
    )


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def plot_variant_delta(
    baseline: List[Dict[str, Any]],
    treated: List[Dict[str, Any]],
    metric_key: str,
    metric_label: Optional[str] = None,
    higher_is_better: bool = True,
) -> go.Figure:
    """Per-model change after treatment, oriented so right is always better."""
    t = tokens()
    base_values = _metric_map(baseline, metric_key)
    treated_values = _metric_map(treated, metric_key)
    rows = [
        (name, treated_values[name] - base_values[name], base_values[name], treated_values[name])
        for name in base_values.keys() & treated_values.keys()
    ]
    if not rows:
        return _empty_figure("No paired models to compare across variants")

    rows.sort(key=lambda row: row[1] * (1 if higher_is_better else -1))
    names = [row[0] for row in rows]
    deltas = [row[1] for row in rows]
    colors = []
    for delta in deltas:
        improved = (delta > 0) == higher_is_better and abs(delta) > 1e-12
        degraded = (delta < 0) == higher_is_better and abs(delta) > 1e-12
        colors.append(t["green"] if improved else t["rose"] if degraded else t["muted"])
    label = metric_label or metric_key.upper().replace("_", " ")

    fig = go.Figure(
        go.Bar(
            y=names,
            x=deltas,
            orientation="h",
            marker=dict(
                color=[_rgba(color, 0.55) for color in colors],
                line=dict(color=colors, width=1.2),
            ),
            text=[f"{delta:+.4f}" for delta in deltas],
            textposition="outside",
            textfont=dict(family=MONO, size=10, color=t["body"]),
            cliponaxis=False,
            customdata=[[row[2], row[3]] for row in rows],
            hovertemplate=(
                "<b>%{y}</b><br>Baseline: %{customdata[0]:.4f}<br>"
                "Treated: %{customdata[1]:.4f}<br>Δ: %{x:+.4f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0.0, line_width=1, line_color=t["axis"])
    fig.update_yaxes(autorange="reversed")
    fig = _polish(
        fig,
        f"Treatment delta · {label} (treated − baseline)",
        height=max(320, 26 * len(names) + 140),
        hovermode="y unified",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Δ " + label, showgrid=True, gridcolor=t["grid"], zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_noise_profile(data_quality: Dict[str, Any]) -> go.Figure:
    """High-frequency noise ratio per feature, baseline vs treated."""
    t = tokens()
    variants = (data_quality or {}).get("variants") or {}
    baseline_cols = (variants.get("baseline") or {}).get("columns") or {}
    treated_cols = (variants.get("treated") or {}).get("columns") or {}
    if not baseline_cols and not treated_cols:
        return _empty_figure("No per-feature noise profile stored for this experiment")

    names = list(dict.fromkeys(list(baseline_cols.keys()) + list(treated_cols.keys())))
    names = [
        name
        for name in names
        if (baseline_cols.get(name) or {}).get("hf_noise_ratio") is not None
        or (treated_cols.get(name) or {}).get("hf_noise_ratio") is not None
    ]
    if not names:
        return _empty_figure("No numeric features were profiled")

    before = [(baseline_cols.get(name) or {}).get("hf_noise_ratio") for name in names]
    after = [(treated_cols.get(name) or {}).get("hf_noise_ratio") for name in names]
    outlier = [
        (baseline_cols.get(name) or {}).get("outlier_iqr_pct")
        if (baseline_cols.get(name) or {}).get("outlier_iqr_pct") is not None
        else (treated_cols.get(name) or {}).get("outlier_iqr_pct")
        for name in names
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names,
            y=before,
            name="Before treatment",
            marker=dict(
                color=_rgba(t["blue"], 0.45),
                line=dict(color=t["blue"], width=1.1),
            ),
            customdata=outlier,
            hovertemplate=(
                "<b>%{x}</b><br>HF noise ratio: %{y:.3f}<br>"
                "IQR outliers: %{customdata}%<extra>Before</extra>"
            ),
        )
    )
    if any(value is not None for value in after):
        fig.add_trace(
            go.Bar(
                x=names,
                y=after,
                name="After treatment",
                marker=dict(
                    color=_rgba(t["green"], 0.5),
                    line=dict(color=t["green"], width=1.1),
                ),
                hovertemplate="<b>%{x}</b><br>HF noise ratio: %{y:.3f}<extra>After</extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(title_text="HF noise ratio")
    return _polish(
        fig,
        "Noise profile · high-frequency energy per feature",
        height=420,
        hovermode="x unified",
        showlegend=True,
    )


def plot_column_distribution(
    histogram: Dict[str, Any], column: str
) -> go.Figure:
    """Overlaid before/after histogram for one treated feature."""
    t = tokens()
    if not histogram or not histogram.get("centers"):
        return _empty_figure("No distribution summary stored for this column")

    centers = histogram.get("centers") or []
    before = histogram.get("before") or []
    after = histogram.get("after")
    widths = [
        (centers[i + 1] - centers[i]) * 0.92 if i + 1 < len(centers) else
        (centers[i] - centers[i - 1]) * 0.92 if i > 0 else 0.9
        for i in range(len(centers))
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=before,
            width=widths,
            name="Before treatment",
            marker=dict(color=_rgba(t["blue"], 0.45), line=dict(color=t["blue"], width=1)),
            hovertemplate="%{x:.4f}<br>Count: %{y}<extra>Before</extra>",
        )
    )
    if after:
        fig.add_trace(
            go.Bar(
                x=centers,
                y=after,
                width=widths,
                name="After treatment",
                marker=dict(color=_rgba(t["green"], 0.45), line=dict(color=t["green"], width=1)),
                hovertemplate="%{x:.4f}<br>Count: %{y}<extra>After</extra>",
            )
        )
    fig.update_layout(barmode="overlay", bargap=0.05)
    fig.update_xaxes(title_text=column)
    fig.update_yaxes(title_text="Rows")
    return _polish(
        fig,
        f"Distribution shift · {column}",
        height=360,
        hovermode="x unified",
        showlegend=True,
    )
