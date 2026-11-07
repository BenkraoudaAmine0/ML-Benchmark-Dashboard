"""
Plotly charts for the ML2 Model Benchmark Studio.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


DARK_TEMPLATE = "plotly_dark"
COLOR_CLASSICAL = "#60A5FA"  # Blue
COLOR_ENSEMBLE = "#22D3EE"   # Cyan
COLOR_ACCENT = "#FBBF24"     # Amber


def plot_metric_comparison(
    leaderboard: List[Dict[str, Any]],
    metric_key: str,
    metric_label: Optional[str] = None,
    higher_is_better: bool = True,
) -> go.Figure:
    """Create a bar chart comparing a specific metric across models."""
    names = []
    values = []
    categories = []
    colors = []

    for item in leaderboard:
        m_val = item["metrics"].get(metric_key)
        if m_val is not None:
            names.append(item["model_name"])
            values.append(m_val)
            cat = item["category"]
            categories.append(cat)
            colors.append(COLOR_ENSEMBLE if cat == "ensemble" else COLOR_CLASSICAL)

    # Sort
    idx_sorted = np.argsort(values)
    if higher_is_better:
        idx_sorted = idx_sorted[::-1]

    s_names = [names[i] for i in idx_sorted]
    s_values = [values[i] for i in idx_sorted]
    s_colors = [colors[i] for i in idx_sorted]

    fig = go.Figure(
        go.Bar(
            x=s_names,
            y=s_values,
            marker=dict(color=s_colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{v:.4f}" for v in s_values],
            textposition="auto",
        )
    )

    label = metric_label or metric_key.upper().replace("_", " ")
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=f"Model Comparison: {label}", font=dict(size=16)),
        xaxis=dict(title="", tickangle=-25),
        yaxis=dict(title=label),
        margin=dict(l=40, r=40, t=50, b=80),
        height=380,
    )
    return fig


def plot_latency_comparison(leaderboard: List[Dict[str, Any]]) -> go.Figure:
    """Create a horizontal bar chart of training times."""
    names = [item["model_name"] for item in leaderboard]
    times = [item["train_time_sec"] for item in leaderboard]
    cats = [item["category"] for item in leaderboard]
    colors = [COLOR_ENSEMBLE if c == "ensemble" else COLOR_CLASSICAL for c in cats]

    # Sort fastest to slowest
    idx_sorted = np.argsort(times)
    s_names = [names[i] for i in idx_sorted]
    s_times = [times[i] for i in idx_sorted]
    s_colors = [colors[i] for i in idx_sorted]

    fig = go.Figure(
        go.Bar(
            y=s_names,
            x=s_times,
            orientation="h",
            marker=dict(color=s_colors),
            text=[f"{t:.4f}s" for t in s_times],
            textposition="auto",
        )
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text="Training Duration (Lower is Faster)", font=dict(size=16)),
        xaxis=dict(title="Execution Time (seconds)"),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    return fig


def plot_radar_metrics(leaderboard: List[Dict[str, Any]], task_type: str) -> go.Figure:
    """Create a radar chart comparing models across multiple metrics."""
    fig = go.Figure()

    if task_type == "classification":
        metric_keys = ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"]
        labels = ["Accuracy", "Balanced Acc", "F1 (Macro)", "Precision", "Recall"]
    else:
        metric_keys = ["r2", "explained_variance"]
        labels = ["R² Score", "Explained Variance"]

    for item in leaderboard:
        vals = []
        for mk in metric_keys:
            val = item["metrics"].get(mk)
            vals.append(val if val is not None else 0.0)
        # Close radar loop
        vals.append(vals[0])

        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=labels + [labels[0]],
                fill="toself",
                name=item["model_name"],
                opacity=0.6,
            )
        )

    fig.update_layout(
        template=DARK_TEMPLATE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1] if task_type == "classification" else None)
        ),
        title=dict(text="Multi-Metric Radar Comparison", font=dict(size=16)),
        margin=dict(l=40, r=40, t=50, b=40),
        height=420,
    )
    return fig


def plot_confusion_matrix(cm: List[List[int]], class_names: Optional[List[str]] = None) -> go.Figure:
    """Plot an interactive confusion matrix heatmap."""
    cm_arr = np.array(cm)
    n_classes = cm_arr.shape[0]
    labels = class_names or [f"Class {i}" for i in range(n_classes)]

    # Normalized matrix
    cm_norm = cm_arr.astype(float) / np.maximum(1, cm_arr.sum(axis=1, keepdims=True))

    annotations = []
    for i in range(n_classes):
        for j in range(n_classes):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{cm_arr[i, j]}</b><br>({cm_norm[i, j]:.1%})",
                    showarrow=False,
                    font=dict(color="white" if cm_norm[i, j] > 0.4 else "#94A3B8", size=12),
                )
            )

    fig = go.Figure(
        go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            colorscale="Viridis",
            showscale=True,
            zmin=0,
            zmax=1,
        )
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text="Confusion Matrix", font=dict(size=16)),
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="Actual Label", autorange="reversed"),
        annotations=annotations,
        margin=dict(l=60, r=40, t=50, b=60),
        height=420,
    )
    return fig


def plot_residuals(actual: List[float], predicted: List[float]) -> go.Figure:
    """Plot Actual vs Predicted scatter plot with residuals for regression."""
    act = np.array(actual)
    pred = np.array(predicted)

    min_val = min(np.min(act), np.min(pred))
    max_val = max(np.max(act), np.max(pred))

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=act,
            y=pred,
            mode="markers",
            name="Predictions",
            marker=dict(color="#22D3EE", size=8, opacity=0.8, line=dict(color="white", width=0.5)),
        )
    )

    # Perfect prediction diagonal line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Fit (y = x)",
            line=dict(color="#FBBF24", dash="dash", width=2),
        )
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text="Actual vs Predicted Values", font=dict(size=16)),
        xaxis=dict(title="Actual Target"),
        yaxis=dict(title="Predicted Target"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=400,
    )
    return fig


def plot_feature_importances(importances: Dict[str, float], top_n: int = 15) -> go.Figure:
    """Plot horizontal bar chart of top feature importances."""
    if not importances:
        fig = go.Figure()
        fig.update_layout(template=DARK_TEMPLATE, title="Feature importances not available for this model.")
        return fig

    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [k for k, v in sorted_items][::-1]
    scores = [v for k, v in sorted_items][::-1]

    fig = go.Figure(
        go.Bar(
            y=features,
            x=scores,
            orientation="h",
            marker=dict(color="#22D3EE"),
            text=[f"{s:.4f}" for s in scores],
            textposition="auto",
        )
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=f"Top {min(top_n, len(features))} Feature Importances", font=dict(size=16)),
        xaxis=dict(title="Importance Weight"),
        yaxis=dict(title=""),
        margin=dict(l=100, r=40, t=50, b=40),
        height=max(350, len(features) * 26),
    )
    return fig

