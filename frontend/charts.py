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

CHART_THEMES: Dict[str, Dict[str, str]] = {
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
        "heat_scale": [
            [0.0, "#F1F5F9"],
            [0.45, "#7C3AED"],
            [1.0, "#0891B2"],
        ],
    },
}

_CLASSICAL = "blue"
_ENSEMBLE = "cyan"
_ACCENT = "amber"


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


def plot_radar_metrics(leaderboard: List[Dict[str, Any]], task_type: str) -> go.Figure:
    """Radar chart comparing models across several metrics."""
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

    palette = [t[_ENSEMBLE], t[_CLASSICAL], t["violet"], t["amber"], t["green"],
               t["pink"], t["orange"], t["rose"]]

    fig = go.Figure()
    for index, item in enumerate(leaderboard):
        values = [item["metrics"].get(key) or 0.0 for key in metric_keys]
        values.append(values[0])
        color = palette[index % len(palette)]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor=_rgba(color, 0.12 if theme() == "dark" else 0.07),
                name=item["model_name"],
                line=dict(color=color, width=2),
                hovertemplate="<b>%{theta}</b><br>%{r:.4f}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=radial_range,
                gridcolor=t["grid"],
                linecolor=t["axis"],
                tickfont=dict(size=9, color=t["muted"], family=MONO),
                angle=90,
            ),
            angularaxis=dict(
                gridcolor=t["grid"],
                linecolor=t["axis"],
                tickfont=dict(size=9, color=t["body"], family=MONO),
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
