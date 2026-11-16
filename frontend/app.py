"""
ML2 Model Benchmark Studio — Streamlit Dashboard Application
============================================================
A modern interactive benchmark dashboard for Classical & Ensemble ML models.
Includes SVM, KNN, Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM, and CatBoost.
Persists all runs in SQLite.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

# Suppress library deprecation warnings from terminal
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    AI_MODEL,
    AI_PROMPT_VERSION,
    CATEGORICAL_IMPUTERS,
    CATEGORICAL_IMPUTER_LABELS,
    DATASETS_DIR,
    INJECTION_METHODS,
    INJECTION_METHOD_LABELS,
    NAN_CAPABLE_MODELS,
    NOISE_STRATEGIES,
    NOISE_STRATEGY_LABELS,
    NUMERIC_IMPUTERS,
    NUMERIC_IMPUTER_LABELS,
    OUTLIER_ACTIONS,
    OUTLIER_ACTION_LABELS,
    OUTLIER_METHODS,
    OUTLIER_METHOD_LABELS,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    VARIANT_BASELINE,
    VARIANT_LABELS,
    VARIANT_TREATED,
)
from core.ai_insights import (
    AIInsightError,
    PROVIDERS,
    generate_report,
    report_to_markdown,
    resolve_settings,
    test_connection,
)
from core.data_loader import (
    detect_task_type,
    discover_datasets,
    load_dataset,
)
from core.data_quality import compare_variants, primary_label_for, primary_metric_for
from core.noise import InjectionConfig, NoiseConfig, OutlierConfig
from core.pipeline import TrainingPipeline
from core.splits import preview_fingerprint
from core.storage import (
    default_variant,
    delete_experiment,
    export_leaderboard_df,
    get_ai_insight,
    get_all_experiments,
    get_benchmark_history,
    get_experiment,
    get_leaderboard,
    get_variants,
    init_db,
    list_splits,
    save_ai_insight,
)
from frontend import charts
from frontend import ui_theme as ui
from frontend.insight_context import CHART_LOCATIONS, build_insight_context
from models.registry import MODEL_REGISTRY, list_available_models


# Page Configuration
_FAVICON = Path(__file__).resolve().parent / "assets" / "favicon.png"
st.set_page_config(
    page_title="ML2 Model Benchmark Studio",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve the active theme, inject the design system and sync the chart palette.
THEME_MODE = ui.requested_mode()
THEME = ui.resolve_theme()
ui.inject_css(THEME_MODE)
charts.set_theme(THEME)
init_db()

# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
if "active_pipeline" not in st.session_state:
    st.session_state.active_pipeline = None
if "selected_exp_id" not in st.session_state:
    st.session_state.selected_exp_id = None


def _shorten(text: object, limit: int = 24) -> str:
    """Trim long dataset / run names so KPI cards stay on one or two lines."""
    value = str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _format_metric(value: float) -> str:
    """Readable number for metric tables: fixed decimals, scientific when extreme."""
    magnitude = abs(float(value))
    if magnitude and (magnitude >= 1_000_000 or magnitude < 1e-3):
        return f"{value:.4g}"
    return f"{value:,.4f}"


def _variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, str(variant).capitalize())


def _variant_selector(experiment_id: int, key: str) -> Optional[str]:
    """Variant picker for experiments that trained both baseline and treated."""
    variants = get_variants(experiment_id)
    if not variants:
        return None
    if len(variants) == 1:
        return variants[0]
    preferred = default_variant(experiment_id) or variants[0]
    return st.selectbox(
        "Data variant",
        options=variants,
        index=variants.index(preferred),
        format_func=_variant_label,
        key=key,
        help="The same split trained twice: untreated (or synthetically noisy) and treated.",
    )


def _json_field(record: Optional[Dict[str, object]], name: str) -> dict:
    if not record:
        return {}
    value = record.get(name)
    if isinstance(value, dict):
        return value
    try:
        import json

        return json.loads(value or "{}")
    except Exception:
        return {}


def _quality_summary(data_quality: dict) -> Dict[str, object]:
    treatment = data_quality.get("treatment") or {}
    noise = treatment.get("noise") or {}
    outliers = treatment.get("outliers") or {}
    injection = treatment.get("injection") or data_quality.get("injection") or {}
    return {
        "noise": noise,
        "outliers": outliers,
        "injection": injection,
        "columns": data_quality.get("columns") or {},
        "variants": data_quality.get("variants") or {},
        "target": data_quality.get("target") or {},
    }


def _quality_columns_df(data_quality: dict, variant: str) -> pd.DataFrame:
    """Per-feature quality table for one variant."""
    columns = (
        ((data_quality.get("variants") or {}).get(variant) or {}).get("columns")
        or data_quality.get("columns")
        or {}
    )
    rows = []
    for column, stats in columns.items():
        rows.append(
            {
                "Feature": column,
                "Missing %": stats.get("missing_pct"),
                "IQR outliers %": stats.get("outlier_iqr_pct"),
                "Z outliers %": stats.get("outlier_z_pct"),
                "Skew": stats.get("skew"),
                "HF noise ratio": stats.get("hf_noise_ratio"),
                "Roughness": stats.get("roughness"),
            }
        )
    return pd.DataFrame(rows)


def _cached_ai_report(experiment_id: int, model: Optional[str] = None) -> Optional[dict]:
    try:
        record = get_ai_insight(experiment_id, AI_PROMPT_VERSION, model)
        return record.get("content") if record else None
    except Exception:
        return None


def _ai_caption(report: Optional[dict], chart_key: str) -> str:
    if not report:
        return ""
    explanation = (report.get("chart_explanations") or {}).get(chart_key)
    return f"<strong>AI read</strong> · {explanation}" if explanation else ""


# ------------------------------------------------------------------------------
# Sidebar Navigation & Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    ui.sidebar_brand()
    ui.theme_switcher()

    st.markdown(
        ui.section("Navigate", "Seven focused workspaces."),
        unsafe_allow_html=True,
    )
    menu = st.radio(
        "Navigation",
        [
            "🏆 Comparison Dashboard",
            "🚀 Train New Models",
            "🧪 Noise & Data Quality",
            "🔬 Model Deep Dive",
            "⚡ Live Inference",
            "🗄️ Experiment History",
            "🤖 AI Interpretation",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown(
        ui.section("Workspace", "Stored benchmarks and the active experiment."),
        unsafe_allow_html=True,
    )
    all_experiments = get_all_experiments()
    st.caption(f"**SQLite storage** · `{len(all_experiments)}` experiments recorded")
    if all_experiments:
        exp_options = {e["id"]: f"#{e['id']} · {e['run_name']} ({e['task_type']})" for e in all_experiments}
        # Default to the newest experiment that actually has results: an empty
        # or in-progress run would otherwise land on a chartless dashboard.
        with_results = [e for e in all_experiments if (e.get("model_count") or 0) > 0]
        default_id = (with_results or all_experiments)[0]["id"]
        current_id = st.session_state.selected_exp_id or default_id
        if current_id not in exp_options:
            current_id = default_id

        selected_id = st.selectbox(
            "Active Experiment",
            options=list(exp_options.keys()),
            format_func=lambda x: exp_options[x],
            index=list(exp_options.keys()).index(current_id),
        )
        st.session_state.selected_exp_id = selected_id

    # Signature layers: frosted menu toggle, background particles, chart motion.
    ui.sidebar_toggle(THEME)
    ui.background_layer(THEME)
    ui.chart_motion()


# ------------------------------------------------------------------------------
# TAB 1: Comparison Dashboard (DEFAULT VIEW)
# ------------------------------------------------------------------------------
if menu == "🏆 Comparison Dashboard":
    st.markdown(
        ui.hero(
            "Benchmark Console",
            "Compare classical baselines with modern boosting and bagging ensemble architectures "
            "under identical, reproducible conditions.",
            meta=["12 architectures", "SQLite run history", "Interactive Plotly charts", "Live inference"],
            eyebrow=f"benchmark console · {THEME} mode",
        ),
        unsafe_allow_html=True,
    )

    all_experiments = get_all_experiments()

    if not all_experiments:
        st.info("👋 Welcome! No experiment benchmarks found yet in the SQLite database. Go to **🚀 Train New Models** to run your first benchmark suite!")
    else:
        exp_id = st.session_state.selected_exp_id or all_experiments[0]["id"]
        exp_info = get_experiment(exp_id)
        dashboard_variant = _variant_selector(exp_id, "dash_variant")
        leaderboard = get_leaderboard(exp_id, variant=dashboard_variant)

        if not leaderboard:
            st.warning("No model results found for this experiment.")
        else:
            task_type = exp_info["task_type"]
            data_quality = _json_field(exp_info, "data_quality_json")
            variant_list = get_variants(exp_id)
            baseline_lb = (
                get_leaderboard(exp_id, variant=VARIANT_BASELINE)
                if len(variant_list) > 1
                else []
            )
            treated_lb = (
                get_leaderboard(exp_id, variant=VARIANT_TREATED)
                if VARIANT_TREATED in variant_list
                else []
            )

            show_ai_notes = st.checkbox(
                "Show cached AI commentary on charts",
                value=False,
                help="Uses the report already generated on the AI Interpretation page; no additional API call.",
            )
            cached_report = _cached_ai_report(exp_id) if show_ai_notes else None

            # Compute Best Model KPI
            if task_type == TASK_CLASSIFICATION:
                primary_metric = "f1_macro"
                primary_label = "F1-Score (Macro)"
                best_model = max(leaderboard, key=lambda x: x["metrics"].get(primary_metric, 0))
                best_score = best_model["metrics"].get(primary_metric, 0)
                best_score_str = f"{best_score:.4f}"
            else:
                primary_metric = "r2"
                primary_label = "R² Score"
                best_model = max(leaderboard, key=lambda x: x["metrics"].get(primary_metric, -999))
                best_score = best_model["metrics"].get(primary_metric, 0)
                best_score_str = f"{best_score:.4f}"

            fastest_model = min(leaderboard, key=lambda x: x["train_time_sec"])

            # Top KPI cards
            st.markdown(
                ui.kpi_grid(
                    [
                        {
                            "label": "Top performer",
                            "value": best_model["model_name"],
                            "hint": f"Best on {primary_label}",
                            "tone": "warn",
                        },
                        {
                            "label": f"Top {primary_label}",
                            "value": best_score_str,
                            "hint": f"Across {len(leaderboard)} models",
                            "tone": "info",
                        },
                        {
                            "label": "Fastest trainer",
                            "value": fastest_model["model_name"],
                            "hint": f"{fastest_model['train_time_sec']:.4f}s latency",
                            "tone": "good",
                        },
                        {
                            "label": "Dataset & task",
                            "value": _shorten(exp_info["dataset_name"]),
                            "hint": f"{task_type.capitalize()} · {exp_info['n_train']} train / {exp_info['n_test']} test",
                            "tone": "neutral",
                        },
                    ],
                    columns=4,
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                ui.section("Leaderboard", "Every model in the active experiment, ranked by the primary metric."),
                unsafe_allow_html=True,
            )

            # Tabular Leaderboard (filtered to the selected variant)
            df_leaderboard = export_leaderboard_df(exp_id, variant=dashboard_variant)
            if not df_leaderboard.empty:
                # Highlight top row
                st.dataframe(
                    df_leaderboard.sort_values(by=primary_metric.upper(), ascending=False),
                    width="stretch",
                    hide_index=True,
                )

            # Interactive Plotly charts
            st.markdown(
                ui.section("Comparative visualizations", "Metric bars and training-time overhead."),
                unsafe_allow_html=True,
            )
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                if task_type == TASK_CLASSIFICATION:
                    metric_choices = ["f1_macro", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro"]
                    default_m = "f1_macro"
                else:
                    metric_choices = ["r2", "rmse", "mae", "explained_variance"]
                    default_m = "r2"

                selected_metric = st.selectbox(
                    "Compare by Metric",
                    metric_choices,
                    index=0,
                    format_func=lambda x: x.upper().replace("_", " "),
                )
                higher_better = selected_metric not in ["rmse", "mae", "mse", "median_ae", "max_error", "log_loss"]
                fig_metric = charts.plot_metric_comparison(
                    leaderboard, selected_metric, higher_is_better=higher_better
                )
                with st.container(key="frame3d-metric"):
                    st.plotly_chart(fig_metric, config=charts.PLOTLY_CONFIG, width="stretch")
                if cached_report:
                    st.markdown(
                        ui.chart_description(
                            _ai_caption(cached_report, "metric_comparison")
                        ),
                        unsafe_allow_html=True,
                    )

            with chart_col2:
                fig_time = charts.plot_latency_comparison(leaderboard)
                with st.container(key="frame3d-latency"):
                    st.plotly_chart(fig_time, config=charts.PLOTLY_CONFIG, width="stretch")
                if cached_report:
                    st.markdown(
                        ui.chart_description(_ai_caption(cached_report, "latency")),
                        unsafe_allow_html=True,
                    )

            # Radar & category trade-off
            st.markdown(
                ui.section("Multi-metric comparison", "Radar overlay and architecture-level summary."),
                unsafe_allow_html=True,
            )
            ranked_models = sorted(
                leaderboard,
                key=lambda item: item["metrics"].get(primary_metric, 0) or 0,
                reverse=True,
            )
            ranked_names = [item["model_name"] for item in ranked_models]

            radar_col1, radar_col2 = st.columns([1.2, 1])

            with radar_col1:
                radar_selection = st.multiselect(
                    "Models in radar",
                    options=ranked_names,
                    default=ranked_names[: min(5, len(ranked_names))],
                    max_selections=8,
                    help="Capped at 8 models so every architecture keeps its own hue.",
                )
                if radar_selection:
                    radar_subset = [
                        item for item in ranked_models if item["model_name"] in radar_selection
                    ]
                    fig_radar = charts.plot_radar_metrics(
                        radar_subset,
                        task_type,
                        colors=charts.model_palette(ranked_names),
                    )
                    with st.container(key="frame3d-radar"):
                        st.plotly_chart(fig_radar, config=charts.PLOTLY_CONFIG, width="stretch")
                else:
                    st.info("Select at least one model to draw the radar.")

            with radar_col2:
                with st.container(key="frame3d-category"):
                    st.plotly_chart(
                        charts.plot_category_aggregate(
                            leaderboard, primary_metric, primary_label
                        ),
                        config=charts.PLOTLY_CONFIG,
                        width="stretch",
                    )
                classical_models = [m for m in leaderboard if m["category"] == "classical"]
                ensemble_models = [m for m in leaderboard if m["category"] == "ensemble"]

                if classical_models and ensemble_models:
                    c_avg_score = np.mean([m["metrics"].get(primary_metric, 0) for m in classical_models])
                    e_avg_score = np.mean([m["metrics"].get(primary_metric, 0) for m in ensemble_models])
                    c_avg_time = np.mean([m["train_time_sec"] for m in classical_models])
                    e_avg_time = np.mean([m["train_time_sec"] for m in ensemble_models])

                    lead = (
                        f"Ensembles lead by +{e_avg_score - c_avg_score:.4f}"
                        if e_avg_score >= c_avg_score
                        else f"Classical models lead by +{c_avg_score - e_avg_score:.4f}"
                    )
                    st.markdown(
                        ui.callout(
                            f"<strong>Classical baselines</strong> · mean {primary_label}: "
                            f"<code>{c_avg_score:.4f}</code> · mean latency <code>{c_avg_time:.4f}s</code><br>"
                            f"<strong>Ensemble architectures</strong> · mean {primary_label}: "
                            f"<code>{e_avg_score:.4f}</code> · mean latency <code>{e_avg_time:.4f}s</code><br>"
                            f"<strong>{lead}</strong> on {primary_label}.",
                            "info",
                        ),
                        unsafe_allow_html=True,
                    )

            # Cost vs. quality trade-off
            st.markdown(
                ui.section("Cost vs. quality", "Score against training time, with the Pareto frontier."),
                unsafe_allow_html=True,
            )
            with st.container(key="frame3d-tradeoff"):
                st.plotly_chart(
                    charts.plot_score_vs_time(
                        leaderboard, primary_metric, primary_label, higher_is_better=True
                    ),
                    config=charts.PLOTLY_CONFIG,
                    width="stretch",
                )
            st.markdown(
                ui.chart_description(
                    "Every point is one architecture. The dotted "
                    "<strong>Pareto frontier</strong> links the models that nothing else beats on "
                    f"both {primary_label} and training time — anything below-right of the line is "
                    "strictly dominated."
                ),
                unsafe_allow_html=True,
            )
            if cached_report:
                st.markdown(
                    ui.chart_description(_ai_caption(cached_report, "score_vs_time")),
                    unsafe_allow_html=True,
                )

            # Benchmark matrix
            st.markdown(
                ui.section("Benchmark matrix", "Rank per metric across every architecture."),
                unsafe_allow_html=True,
            )
            with st.container(key="frame3d-heatmap"):
                st.plotly_chart(
                    charts.plot_metric_heatmap(leaderboard, task_type),
                    config=charts.PLOTLY_CONFIG,
                    width="stretch",
                )
            st.markdown(
                ui.chart_description(
                    "Each cell is the model's rank on that metric — <strong>#1 is best</strong>. "
                    "Lower-is-better metrics (loss, error) are ranked in the correct direction; "
                    "hover a cell for the raw value, or read the leaderboard above for all figures."
                ),
                unsafe_allow_html=True,
            )
            if cached_report:
                st.markdown(
                    ui.chart_description(_ai_caption(cached_report, "heatmap")),
                    unsafe_allow_html=True,
                )

            # Head-to-head comparison
            st.markdown(
                ui.section("Head-to-head matchup", "Metric-by-metric delta between two models."),
                unsafe_allow_html=True,
            )
            h2h_c1, h2h_c2 = st.columns(2)
            model_names = [m["model_name"] for m in leaderboard]
            with h2h_c1:
                m1_name = st.selectbox("Model A", model_names, index=0)
            with h2h_c2:
                m2_name = st.selectbox("Model B", model_names, index=min(1, len(model_names) - 1))

            m1_data = next(m for m in leaderboard if m["model_name"] == m1_name)
            m2_data = next(m for m in leaderboard if m["model_name"] == m2_name)

            diff_records = []
            for k in m1_data["metrics"].keys():
                v1 = m1_data["metrics"].get(k)
                v2 = m2_data["metrics"].get(k)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    delta = v1 - v2
                    diff_records.append(
                        {
                            "Metric": k.upper().replace("_", " "),
                            f"{m1_name}": f"{v1:.4f}",
                            f"{m2_name}": f"{v2:.4f}",
                            "Delta (A - B)": f"{delta:+.4f}",
                        }
                    )
            diff_records.append(
                {
                    "Metric": "TRAINING TIME (S)",
                    f"{m1_name}": f"{m1_data['train_time_sec']:.4f}s",
                    f"{m2_name}": f"{m2_data['train_time_sec']:.4f}s",
                    "Delta (A - B)": f"{(m1_data['train_time_sec'] - m2_data['train_time_sec']):+.4f}s",
                }
            )
            with st.container(key="frame3d-headtohead"):
                st.plotly_chart(
                    charts.plot_head_to_head(m1_data, m2_data, task_type),
                    config=charts.PLOTLY_CONFIG,
                    width="stretch",
                )
            st.markdown(
                ui.chart_description(
                    "Bars show the <strong>relative advantage of Model A over Model B</strong> per "
                    "metric — the gap divided by the larger of the two values, so different scales "
                    "stay comparable and one extreme metric cannot dominate. Bars to the right mean "
                    "A wins; error metrics, overfitting gap and training time are flipped to the "
                    "same direction."
                ),
                unsafe_allow_html=True,
            )
            if cached_report:
                st.markdown(
                    ui.chart_description(_ai_caption(cached_report, "head_to_head")),
                    unsafe_allow_html=True,
                )
            with st.expander("Exact values", expanded=False):
                st.table(pd.DataFrame(diff_records))

            # --------------------------------------------------------------
            # Noise ablation (only when the run trained both variants)
            # --------------------------------------------------------------
            if baseline_lb and treated_lb:
                st.markdown(
                    ui.section(
                        "Noise ablation",
                        "Identical split and models, trained on untreated and treated data.",
                    ),
                    unsafe_allow_html=True,
                )
                comparison = compare_variants(
                    baseline_lb, treated_lb, primary_metric, higher_is_better=True
                )
                trend = "improved" if comparison["mean_delta"] > 0 else "degraded"
                best_change = (
                    f"{comparison['baseline_best']['model']} → {comparison['treated_best']['model']}"
                    if comparison["best_changed"]
                    else comparison["baseline_best"]["model"] or "—"
                )
                st.markdown(
                    ui.kpi_grid(
                        [
                            {
                                "label": "Mean Δ after treatment",
                                "value": f"{comparison['mean_delta']:+.4f}",
                                "hint": f"{trend} on {primary_label}",
                                "tone": "good" if comparison["mean_delta"] > 0 else "danger",
                            },
                            {
                                "label": "Wins / losses",
                                "value": f"{comparison['wins']} / {comparison['losses']}",
                                "hint": f"{comparison['unchanged']} unchanged",
                                "tone": "info",
                            },
                            {
                                "label": "Best model",
                                "value": _shorten(best_change, 30),
                                "hint": "Baseline → treated",
                                "tone": "warn",
                            },
                            {
                                "label": "Rows removed",
                                "value": str(int(_quality_summary(data_quality)["outliers"].get("removed_rows") or 0)),
                                "hint": "Training split only",
                                "tone": "neutral",
                            },
                        ],
                        columns=4,
                    ),
                    unsafe_allow_html=True,
                )
                ablation_col1, ablation_col2 = st.columns(2)
                with ablation_col1:
                    with st.container(key="frame3d-variant-compare"):
                        st.plotly_chart(
                            charts.plot_variant_comparison(
                                baseline_lb,
                                treated_lb,
                                primary_metric,
                                primary_label,
                                higher_is_better=True,
                            ),
                            config=charts.PLOTLY_CONFIG,
                            width="stretch",
                        )
                with ablation_col2:
                    with st.container(key="frame3d-variant-delta"):
                        st.plotly_chart(
                            charts.plot_variant_delta(
                                baseline_lb,
                                treated_lb,
                                primary_metric,
                                primary_label,
                                higher_is_better=True,
                            ),
                            config=charts.PLOTLY_CONFIG,
                            width="stretch",
                        )
                st.markdown(
                    ui.chart_description(
                        "Both variants share the same rows and test set, so every difference "
                        "is caused by the treatment step. The delta chart is oriented so bars "
                        "to the right always mean the treated data scored better."
                    ),
                    unsafe_allow_html=True,
                )
                if cached_report:
                    st.markdown(
                        ui.chart_description(
                            _ai_caption(cached_report, "variant_comparison")
                        ),
                        unsafe_allow_html=True,
                    )
                with st.expander("Ablation values", expanded=False):
                    import pandas as _pd

                    ablation_rows = [
                        {
                            "Model": row["model_name"],
                            "Baseline": row["baseline"],
                            "Treated": row["treated"],
                            "Δ": row["delta"],
                            "% change": row["pct_change"],
                            "Verdict": row["direction"],
                        }
                        for row in comparison["models"]
                    ]
                    st.dataframe(
                        _pd.DataFrame(ablation_rows),
                        width="stretch",
                        hide_index=True,
                    )

            # --------------------------------------------------------------
            # Data quality snapshot
            # --------------------------------------------------------------
            if data_quality:
                with st.expander("Data quality snapshot", expanded=False):
                    summary = _quality_summary(data_quality)
                    columns_profile = summary["columns"]
                    outlier_values = [
                        stats.get("outlier_iqr_pct")
                        for stats in columns_profile.values()
                        if isinstance(stats.get("outlier_iqr_pct"), (int, float))
                    ]
                    mean_outlier = (
                        float(np.mean(outlier_values)) if outlier_values else 0.0
                    )
                    noise_cfg = summary["noise"]
                    injection_cfg = summary["injection"]
                    st.markdown(
                        ui.kpi_grid(
                            [
                                {
                                    "label": "Missing cells",
                                    "value": f"{data_quality.get('missing_pct', 0):.2f}%",
                                    "hint": f"{data_quality.get('missing_cells', 0):,} cells",
                                    "tone": "info",
                                },
                                {
                                    "label": "Mean IQR outliers",
                                    "value": f"{mean_outlier:.2f}%",
                                    "hint": "Across numeric features",
                                    "tone": "warn",
                                },
                                {
                                    "label": "Duplicate rows",
                                    "value": f"{data_quality.get('duplicate_rows', 0):,}",
                                    "hint": f"of {data_quality.get('n_rows', 0):,}",
                                    "tone": "neutral",
                                },
                                {
                                    "label": "Noise treatment",
                                    "value": str(noise_cfg.get("strategy", "none")),
                                    "hint": (
                                        "Injection: " + str(injection_cfg.get("method", "none"))
                                        if injection_cfg.get("enabled")
                                        else "No synthetic noise"
                                    ),
                                    "tone": "good" if noise_cfg.get("enabled") else "neutral",
                                },
                            ],
                            columns=4,
                        ),
                        unsafe_allow_html=True,
                    )
                    with st.container(key="frame3d-noise-profile"):
                        st.plotly_chart(
                            charts.plot_noise_profile(data_quality),
                            config=charts.PLOTLY_CONFIG,
                            width="stretch",
                        )
                    st.markdown(
                        ui.chart_description(
                            "High-frequency energy is the standard deviation of the residual "
                            "after a 5-row rolling median, divided by the column's standard "
                            "deviation. Treatment should lower it; a value that barely moves "
                            "means the smoothing window was too small for that feature."
                        ),
                        unsafe_allow_html=True,
                    )

    st.markdown(ui.footer(), unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 2: Train New Models
# ------------------------------------------------------------------------------
elif menu == "🚀 Train New Models":
    st.markdown(
        ui.hero(
            "Train New Models",
            "Select a dataset, configure the task, choose the architectures, and run the benchmark "
            "pipeline with persistent SQLite storage.",
            meta=["Auto task detection", "Leak-free preprocessing", "12 architectures", "Joblib artifacts"],
            eyebrow=f"benchmark console · {THEME} mode",
        ),
        unsafe_allow_html=True,
    )

    # Step 1: Select or upload dataset
    st.markdown(
        ui.section("01 · Choose dataset", "Repository CSVs or your own upload."),
        unsafe_allow_html=True,
    )
    ds_source_tab1, ds_source_tab2 = st.tabs(["📁 Existing Repository Datasets", "📤 Upload New CSV"])

    df = None
    dataset_name = "Custom Dataset"

    with ds_source_tab1:
        discovered = discover_datasets(DATASETS_DIR)
        if discovered:
            chosen_key = st.selectbox("Select CSV dataset from repository", list(discovered.keys()))
            if chosen_key:
                dataset_name = Path(chosen_key).name
                df = load_dataset(discovered[chosen_key])
                st.caption(f"Loaded: `{chosen_key}` · **Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
        else:
            st.info("No CSV files discovered in Datasets/ folder.")

    with ds_source_tab2:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name
            st.success(f"Uploaded `{dataset_name}`: {df.shape[0]} rows × {df.shape[1]} columns")

    if df is not None:
        total_rows = len(df)
        with st.expander("👀 Preview Dataset (First 5 Rows)", expanded=False):
            st.dataframe(df.head(), width="stretch")

        st.markdown("---")
        # Step 2: Target & task configuration
        st.markdown(
            ui.section("02 · Target column & task", "Automatic inference with a manual override."),
            unsafe_allow_html=True,
        )
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            target_col = st.selectbox("Target Column (y)", df.columns.tolist(), index=len(df.columns) - 1)

        inferred_task = detect_task_type(df[target_col])
        with col_t2:
            task_type = st.radio(
                "Task Type",
                [TASK_CLASSIFICATION, TASK_REGRESSION],
                index=0 if inferred_task == TASK_CLASSIFICATION else 1,
                horizontal=True,
                help="Auto-detected based on target column datatype and cardinality.",
            )

        # Step 3: Dataset size & train/test split
        st.markdown(
            ui.section(
                "03 · Dataset size, split & preprocessing",
                "Split ratio, seed, imputation, noise treatment and split persistence.",
            ),
            unsafe_allow_html=True,
        )
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            test_size = st.slider("Test Split Ratio", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        with col_s2:
            seed = st.number_input("Random Seed", value=42, step=1)
        with col_s3:
            scale_num = st.checkbox("Standardize Numeric Features", value=True)

        col_sub1, col_sub2, col_sub3 = st.columns([1, 1.2, 1.2])
        with col_sub1:
            enable_subsample = st.checkbox(
                "⚡ Limit Dataset Rows",
                value=(total_rows > 25000),
                help="Recommended for large datasets (e.g. 100k rows) to reduce training time to 20k rows or custom size.",
            )
        max_samples_val = None
        sampling_label = "Random Sample"
        with col_sub2:
            if enable_subsample:
                default_sub = min(20000, total_rows)
                max_samples_val = st.number_input(
                    f"Max rows to use (Total: {total_rows:,})",
                    min_value=100,
                    max_value=total_rows,
                    value=default_sub,
                    step=1000,
                )
            else:
                st.caption(f"Using full dataset: **{total_rows:,} rows**.")

        with col_sub3:
            if enable_subsample:
                sampling_label = st.radio(
                    "Selection Method",
                    ["Random Sample", "First N Rows (By Order)"],
                    horizontal=True,
                    help="Choose 'Random Sample' for uniform shuffling or 'First N Rows' to preserve dataset sequence (e.g. time series).",
                )
                if sampling_label == "Random Sample":
                    st.caption(f"⚡ Shuffling randomly using seed `{seed}`.")
                else:
                    st.caption(f"⚡ Taking the first `{max_samples_val:,}` rows in original file order.")
        sampling_strategy = "first" if sampling_label.startswith("First") else "random"
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        all_features = df.columns.tolist()

        # ---- Imputation -------------------------------------------------
        with st.expander("🧩 Imputation", expanded=False):
            imp_col1, imp_col2 = st.columns(2)
            imputer_params: Dict[str, object] = {}
            with imp_col1:
                numeric_imputer = st.selectbox(
                    "Numeric imputation",
                    options=NUMERIC_IMPUTERS,
                    index=NUMERIC_IMPUTERS.index("median"),
                    format_func=lambda key: NUMERIC_IMPUTER_LABELS.get(key, key),
                    help=(
                        "Median is the robust default. KNN uses the k most similar rows "
                        "(features are scaled first). MICE and MissForest are model-based. "
                        "Interpolation / forward fill suit ordered data. 'None' only works "
                        "with XGBoost, LightGBM and CatBoost."
                    ),
                )
                if numeric_imputer == "knn":
                    imputer_params["n_neighbors"] = int(
                        st.number_input("KNN neighbours", min_value=1, max_value=50, value=5, step=1)
                    )
                    imputer_params["weights"] = st.selectbox(
                        "KNN distance weights", ["uniform", "distance"]
                    )
                elif numeric_imputer in ("iterative", "iterative_rf"):
                    imputer_params["max_iter"] = int(
                        st.number_input("MICE iterations", min_value=1, max_value=50, value=10, step=1)
                    )
                    if numeric_imputer == "iterative_rf":
                        imputer_params["rf_estimators"] = int(
                            st.number_input(
                                "Random-forest estimators",
                                min_value=10,
                                max_value=300,
                                value=50,
                                step=10,
                            )
                        )
                        st.caption("⚠️ MissForest-style imputation is significantly slower.")
                elif numeric_imputer == "constant":
                    imputer_params["fill_value"] = st.number_input(
                        "Constant fill value", value=0.0
                    )
                if numeric_imputer != "none":
                    imputer_params["add_indicator"] = st.checkbox(
                        "Add a was-missing indicator column",
                        value=(numeric_imputer == "constant"),
                        help="Lets the model learn that a value was absent.",
                    )
                imputer_params["random_state"] = int(seed)

            with imp_col2:
                categorical_imputer = st.selectbox(
                    "Categorical imputation",
                    options=CATEGORICAL_IMPUTERS,
                    index=0,
                    format_func=lambda key: CATEGORICAL_IMPUTER_LABELS.get(key, key),
                )
                if categorical_imputer == "constant":
                    imputer_params["cat_fill_value"] = st.text_input(
                        "Constant category", value="Missing"
                    )
                st.caption(
                    "**Other algorithms worth knowing:** group-wise (conditional) imputation, "
                    "matrix-factorisation and autoencoder imputation — accurate but heavier; "
                    "model-native NaN handling is available through **None** above."
                )

        # ---- Noise treatment -------------------------------------------
        with st.expander("🌊 Noise treatment & outliers", expanded=False):
            enable_noise = st.checkbox(
                "Enable noise treatment",
                value=False,
                help="Denoise numeric features. Applied to train and test separately, after the split.",
            )
            noise_strategy = "none"
            noise_window = 5
            noise_center = False
            noise_polyorder = 2
            noise_k = 3.0
            noise_columns: List[str] = []
            noise_order_column = None
            if enable_noise:
                noise_col1, noise_col2, noise_col3 = st.columns(3)
                with noise_col1:
                    noise_strategy = st.selectbox(
                        "Denoising strategy",
                        options=[s for s in NOISE_STRATEGIES if s != "none"],
                        format_func=lambda key: NOISE_STRATEGY_LABELS.get(key, key),
                    )
                with noise_col2:
                    if noise_strategy in ("rolling_mean", "rolling_median", "ema", "savgol", "hampel"):
                        noise_window = st.slider("Window size", min_value=2, max_value=51, value=5, step=1)
                    if noise_strategy == "savgol":
                        noise_polyorder = st.slider("Polynomial order", 1, 5, 2)
                    if noise_strategy in ("iqr_clip", "zscore_clip"):
                        noise_k = st.number_input("Fence multiplier k", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
                with noise_col3:
                    if noise_strategy in ("rolling_mean", "rolling_median"):
                        noise_center = st.checkbox("Centre the window", value=False)
                    order_choice = st.selectbox(
                        "Row order for smoothing",
                        ["Original row order"] + all_features,
                        help="Rolling filters need a meaningful sequence. Prefer a timestamp column for time-series data.",
                    )
                    noise_order_column = None if order_choice == "Original row order" else order_choice
                noise_columns = st.multiselect(
                    "Columns to denoise",
                    options=numeric_features,
                    default=[],
                    help="Leave empty to treat every numeric feature.",
                )

            out_col1, out_col2, out_col3 = st.columns(3)
            with out_col1:
                outlier_method = st.selectbox(
                    "Outlier handling",
                    options=OUTLIER_METHODS,
                    index=0,
                    format_func=lambda key: OUTLIER_METHOD_LABELS.get(key, key),
                )
            outlier_action = "clip"
            outlier_k = 1.5
            outlier_contamination = 0.02
            outlier_columns: List[str] = []
            if outlier_method != "none":
                with out_col2:
                    outlier_action = st.selectbox(
                        "Action",
                        options=OUTLIER_ACTIONS,
                        index=0,
                        format_func=lambda key: OUTLIER_ACTION_LABELS.get(key, key),
                        help="Removal only ever drops training rows; the test set stays identical.",
                    )
                with out_col3:
                    if outlier_method in ("iqr", "zscore"):
                        outlier_k = st.number_input(
                            "Outlier threshold k", min_value=0.5, max_value=10.0, value=1.5, step=0.5
                        )
                    else:
                        outlier_contamination = st.slider(
                            "Expected contamination", 0.005, 0.20, 0.02, step=0.005
                        )
                outlier_columns = st.multiselect(
                    "Columns to inspect for outliers",
                    options=numeric_features,
                    default=[],
                )
            if outlier_method == "isolation_forest" and outlier_action == "clip":
                st.caption(
                    "Isolation Forest has no natural fences; flagged cells will be set to "
                    "missing so the imputer handles them."
                )

            st.markdown("---")
            inj_col1, inj_col2, inj_col3 = st.columns(3)
            with inj_col1:
                injection_method = st.selectbox(
                    "Synthetic noise injection",
                    options=INJECTION_METHODS,
                    index=0,
                    format_func=lambda key: INJECTION_METHOD_LABELS.get(key, key),
                    help="Optional: add controlled noise first so a clean dataset still has a 'noisy' side.",
                )
            injection_ratio = 0.05
            injection_spike_ratio = 0.01
            if injection_method != "none":
                with inj_col2:
                    injection_ratio = st.slider(
                        "Noise strength / missing share", 0.001, 0.5, 0.05, step=0.005
                    )
                if injection_method == "spikes":
                    with inj_col3:
                        injection_spike_ratio = st.slider(
                            "Share of spiked cells", 0.001, 0.2, 0.01, step=0.005
                        )
            injection_columns = st.multiselect(
                "Columns to inject noise into",
                options=numeric_features,
                default=[],
                help="Leave empty for every numeric feature.",
            )

            treatment_enabled = enable_noise or outlier_method != "none"
            compare_noise = st.checkbox(
                "Also train the untreated baseline for comparison",
                value=True,
                disabled=not treatment_enabled,
                key="compare_noise_toggle",
                help="Adds a second set of results on the same rows so the ablation is exact.",
            )
            if not treatment_enabled and injection_method == "none":
                st.caption(
                    "No treatment selected: this run produces a single baseline variant. "
                    "Enable denoising or outlier handling to unlock the comparison."
                )

        # ---- Split persistence -----------------------------------------
        with st.expander("💾 Split persistence", expanded=False):
            reuse_split = st.checkbox(
                "Reuse a saved split when the data and settings match",
                value=True,
                help="Content-fingerprinted splits are reloaded, so re-running never changes the partitions.",
            )
            saved_splits = list_splits(dataset_name=dataset_name, target_col=target_col)
            split_options: Dict[int, str] = {0: "Auto · match fingerprint"}
            for record in saved_splits:
                split_options[record["id"]] = (
                    f"#{record['id']} · {record['n_train']:,}/{record['n_test']:,} rows · "
                    f"test {record['test_size']:.2f} · seed {record['random_state']} · "
                    f"{record['created_at']}"
                )
            chosen_split = st.selectbox(
                "Saved split",
                options=list(split_options.keys()),
                format_func=lambda key: split_options[key],
                help=(
                    "Auto reuses the split whose fingerprint matches the current data and "
                    "settings. Pick an explicit split to pin an older partition."
                ),
            )
            forced_split_id = None if chosen_split == 0 else int(chosen_split)
            if forced_split_id:
                st.caption(
                    f"🔒 Split #{forced_split_id} will be loaded; its target column, task, "
                    "test ratio, seed and row limits are reused for this run."
                )
            else:
                reuse_match = preview_fingerprint(
                    df,
                    target_col,
                    test_size=test_size,
                    random_state=int(seed),
                    drop_cols=[],
                    max_samples=int(max_samples_val) if max_samples_val else None,
                    sampling_strategy=sampling_strategy,
                    task=task_type,
                )
                if reuse_match:
                    st.caption(
                        f"♻️ A run will reuse saved split **#{reuse_match['id']}** "
                        f"({reuse_match['n_train']:,} train / {reuse_match['n_test']:,} test)."
                    )
                else:
                    st.caption(
                        "🆕 A new split will be created, fingerprinted and saved so the next "
                        "identical run reuses it."
                    )

        # Step 4: Model selection
        st.markdown(
            ui.section("04 · Select models", "Start from a preset, then fine-tune the selection."),
            unsafe_allow_html=True,
        )
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        registered_models = list_available_models(task_type)
        all_ids = [m["id"] for m in registered_models]
        core_ids = [m["id"] for m in registered_models if m.get("is_core")]
        classic_ids = [m["id"] for m in registered_models if m["category"] == "classical"]
        ensemble_ids = [m["id"] for m in registered_models if m["category"] == "ensemble"]

        if "selected_models" not in st.session_state:
            st.session_state.selected_models = core_ids

        with col_btn1:
            if st.button("The 8 Core Models", width="stretch"):
                st.session_state.selected_models = core_ids
        with col_btn2:
            if st.button("All 12 Models (+ Stacking)", width="stretch"):
                st.session_state.selected_models = all_ids
        with col_btn3:
            if st.button("Classical Only", width="stretch"):
                st.session_state.selected_models = classic_ids
        with col_btn4:
            if st.button("Ensembles Only", width="stretch"):
                st.session_state.selected_models = ensemble_ids

        cols_m = st.columns(4)
        chosen_model_ids = []
        for idx, m in enumerate(registered_models):
            col = cols_m[idx % 4]
            with col:
                is_checked = m["id"] in st.session_state.selected_models
                chk = st.checkbox(
                    f"{m['name']}",
                    value=is_checked,
                    key=f"chk_model_{m['id']}",
                    help=f"Category: {m['category'].capitalize()}",
                )
                if chk:
                    chosen_model_ids.append(m["id"])

        # Step 5: Hyperparameters (Collapsible)
        with st.expander("⚙️ Advanced Hyperparameters (Optional)", expanded=False):
            st.caption("Tune defaults or leave them to automatically use benchmark baselines.")
            custom_hyperparams = {}
            for m in registered_models:
                mid = m["id"]
                if mid in chosen_model_ids:
                    st.markdown(f"**{m['name']}**")
                    if mid == "svm":
                        c_val = st.number_input("SVM C", value=1.0, min_value=0.01, step=0.5, key="svm_c")
                        k_val = st.selectbox("SVM Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_k")
                        custom_hyperparams[mid] = {"C": c_val, "kernel": k_val}
                    elif mid == "knn":
                        k_neighbors = st.number_input("KNN n_neighbors", value=5, min_value=1, step=1, key="knn_k")
                        custom_hyperparams[mid] = {"n_neighbors": int(k_neighbors)}
                    elif mid == "dtree":
                        max_d = st.slider("Decision Tree Max Depth", 1, 30, 10, key="dt_d")
                        custom_hyperparams[mid] = {"max_depth": max_d}
                    elif mid in ["rf", "xgboost", "lightgbm", "catboost"]:
                        n_est = st.slider(f"{m['name']} Estimators / Iterations", 20, 300, 100, key=f"{mid}_est")
                        lr = st.slider(f"{m['name']} Learning Rate", 0.01, 1.0, 0.1, key=f"{mid}_lr") if mid != "rf" else None
                        p = {"n_estimators" if mid != "catboost" else "iterations": n_est}
                        if lr:
                            p["learning_rate"] = lr
                        custom_hyperparams[mid] = p

        st.markdown("---")
        # Step 6: Launch
        run_name = st.text_input("Experiment Run Name", value=f"{Path(dataset_name).stem}_{task_type}")

        if st.button("🚀 Run Training Pipeline", type="primary", width="stretch"):
            if not chosen_model_ids:
                st.error("Please select at least one model to train.")
            elif numeric_imputer == "none" and any(
                mid not in NAN_CAPABLE_MODELS for mid in chosen_model_ids
            ):
                unsupported = [
                    MODEL_REGISTRY[mid].display_name
                    for mid in chosen_model_ids
                    if mid not in NAN_CAPABLE_MODELS
                ]
                st.error(
                    "Numeric imputation is set to **None**, but these models cannot consume "
                    f"NaN values: {', '.join(unsupported)}. Pick an imputation strategy or "
                    "select only XGBoost / LightGBM / CatBoost."
                )
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(mid: str, pct: float, msg: str):
                    progress_bar.progress(pct)
                    status_text.markdown(f"⏳ **Status**: {msg}")

                # Guard: regression requires a numeric (or numeric-coercible) target
                if task_type == TASK_REGRESSION:
                    raw_target = df[target_col]
                    coerced_target = pd.to_numeric(raw_target, errors="coerce")
                    invalid_target = coerced_target.isna() & raw_target.notna()
                    if invalid_target.any():
                        samples = raw_target[invalid_target].astype(str).unique()[:3].tolist()
                        st.error(
                            f"Target column `{target_col}` has {int(invalid_target.sum()):,} "
                            f"non-numeric value(s) (e.g. {samples}), so it cannot be used for "
                            "regression. Pick a numeric target column, or switch **Task Type** "
                            "to classification."
                        )
                        st.stop()

                pipeline = TrainingPipeline(
                    df=df,
                    target_col=target_col,
                    task=task_type,
                    test_size=test_size,
                    random_state=int(seed),
                    scale_numeric=scale_num,
                    max_samples=max_samples_val,
                    dataset_name=dataset_name,
                    sampling_strategy=sampling_strategy,
                    numeric_imputer=numeric_imputer,
                    categorical_imputer=categorical_imputer,
                    imputer_params=imputer_params,
                    noise_config=NoiseConfig(
                        strategy=noise_strategy,
                        columns=noise_columns,
                        window=noise_window,
                        center=noise_center,
                        polyorder=noise_polyorder,
                        k=noise_k,
                        order_column=noise_order_column,
                    ),
                    outlier_config=OutlierConfig(
                        method=outlier_method,
                        action=outlier_action,
                        columns=outlier_columns,
                        k=outlier_k,
                        contamination=outlier_contamination,
                    ),
                    injection_config=InjectionConfig(
                        method=injection_method,
                        columns=injection_columns,
                        ratio=injection_ratio,
                        spike_ratio=injection_spike_ratio,
                        seed=int(seed),
                    ),
                    compare_noise=compare_noise,
                    split_id=forced_split_id,
                    save_split=True,
                    allow_split_reuse=reuse_split,
                )

                try:
                    pipeline.prepare()
                    results = pipeline.run_benchmark(
                        model_ids=chosen_model_ids,
                        custom_params=custom_hyperparams,
                        run_name=run_name,
                        dataset_name=dataset_name,
                        progress_cb=update_progress,
                    )
                    st.session_state.active_pipeline = pipeline
                    st.session_state.selected_exp_id = results["experiment_id"]
                    progress_bar.progress(1.0)
                    status_text.markdown("✅ **Training complete!** Results saved to SQLite.")
                    variant_note = (
                        f" Variants: {' + '.join(results['variants'])}."
                        if results.get("variants")
                        else ""
                    )
                    split_note = (
                        f" Split #{results['split_id']} ({results['split_source']})."
                        if results.get("split_id")
                        else ""
                    )
                    st.success(
                        f"Benchmark finished! {len(results['results'])} model runs completed."
                        f"{variant_note}{split_note} Head to the **🏆 Comparison Dashboard** "
                        "to review the results."
                    )
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during training pipeline: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    st.markdown(ui.footer(), unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 3: Model Deep Dive
# ------------------------------------------------------------------------------
elif menu == "🔬 Model Deep Dive":
    st.markdown(
        ui.hero(
            "Model Deep Dive",
            "Inspect individual model performance, confusion matrices, residuals and feature importances.",
            meta=["Per-model diagnostics", "Overfitting gap", "Feature importances", "Artifact download"],
            eyebrow=f"benchmark console · {THEME} mode",
        ),
        unsafe_allow_html=True,
    )

    all_experiments = get_all_experiments()
    if not all_experiments:
        st.info("No experiment benchmarks found in the database. Run a training first!")
    else:
        exp_id = st.session_state.selected_exp_id or all_experiments[0]["id"]
        exp_info = get_experiment(exp_id)
        deep_variant = _variant_selector(exp_id, "deep_variant")
        leaderboard = get_leaderboard(exp_id, variant=deep_variant)

        if not leaderboard:
            st.warning("No model results available.")
        else:
            model_options = {m["id"]: f"{m['model_name']} ({m['category'].capitalize()})" for m in leaderboard}
            chosen_res_id = st.selectbox(
                "Select Model to Inspect",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
            )
            selected_model = next(m for m in leaderboard if m["id"] == chosen_res_id)
            metrics = selected_model["metrics"]
            task_type = exp_info["task_type"]

            # Model detail cards
            score_str = f"{metrics.get('f1_macro', 0):.4f}" if task_type == TASK_CLASSIFICATION else f"{metrics.get('r2', 0):.4f}"
            st.markdown(
                ui.kpi_grid(
                    [
                        {
                            "label": "Model architecture",
                            "value": selected_model["model_name"],
                            "hint": f"Category: {selected_model['category'].capitalize()}",
                            "tone": "info",
                        },
                        {
                            "label": "Training time",
                            "value": f"{selected_model['train_time_sec']:.4f}s",
                            "hint": "Wall-clock latency",
                            "tone": "neutral",
                        },
                        {
                            "label": "Primary score",
                            "value": score_str,
                            "hint": "F1-Macro" if task_type == TASK_CLASSIFICATION else "R² Score",
                            "tone": "warn",
                        },
                    ],
                    columns=3,
                ),
                unsafe_allow_html=True,
            )

            # Overfitting / Generalization Diagnostic
            if "train_f1_macro" in metrics or "train_r2" in metrics:
                tr_score = metrics.get("train_f1_macro") if task_type == TASK_CLASSIFICATION else metrics.get("train_r2")
                te_score = metrics.get("f1_macro") if task_type == TASK_CLASSIFICATION else metrics.get("r2")
                gap = metrics.get("overfit_gap", 0.0)
                is_overfit = metrics.get("is_overfitting", False)

                verdict = (
                    "<strong>High overfitting risk</strong> — the model has memorised training noise."
                    if is_overfit
                    else "<strong>Well generalised</strong> — the train/test gap is healthy."
                )
                st.markdown(
                    ui.callout(
                        f"<strong>Generalization diagnostic (train vs test)</strong><br>"
                        f"Train score <code>{tr_score:.4f}</code> · test score "
                        f"<code>{te_score:.4f}</code> · gap <code>{gap:+.4f}</code><br>{verdict}",
                        "danger" if is_overfit else "success",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            col_diag1, col_diag2 = st.columns(2)

            with col_diag1:
                if task_type == TASK_CLASSIFICATION:
                    st.markdown(
                        ui.chart_header("Confusion matrix", "row share"),
                        unsafe_allow_html=True,
                    )
                    cm = metrics.get("confusion_matrix")
                    if cm:
                        fig_cm = charts.plot_confusion_matrix(cm)
                        with st.container(key="frame3d-confusion"):
                            st.plotly_chart(fig_cm, config=charts.PLOTLY_CONFIG, width="stretch")
                    else:
                        st.info("Confusion matrix not recorded.")
                else:
                    st.markdown(
                        ui.chart_header("Actual vs predicted", "prediction sample"),
                        unsafe_allow_html=True,
                    )
                    pred_sample = selected_model.get("predictions_sample", {})
                    act = pred_sample.get("actual", [])
                    pred = pred_sample.get("predicted", [])
                    if act and pred:
                        fig_res = charts.plot_residuals(act, pred)
                        with st.container(key="frame3d-residuals"):
                            st.plotly_chart(fig_res, config=charts.PLOTLY_CONFIG, width="stretch")
                    else:
                        st.info("Predictions sample not available.")

            with col_diag2:
                st.markdown(
                    ui.chart_header("Feature importances", "top 12"),
                    unsafe_allow_html=True,
                )
                fi = selected_model.get("feature_importances", {})
                if fi:
                    fig_fi = charts.plot_feature_importances(fi, top_n=12)
                    with st.container(key="frame3d-importance"):
                        st.plotly_chart(fig_fi, config=charts.PLOTLY_CONFIG, width="stretch")
                else:
                    st.info("Feature importances not natively available for this model type (e.g. distance-based KNN or RBF SVM).")

            # Performance report: a formatted table for both task types
            st.markdown(
                ui.section(
                    "Performance report",
                    "Per-class precision / recall / support, or the full regression metric suite.",
                ),
                unsafe_allow_html=True,
            )
            if task_type == TASK_CLASSIFICATION:
                clf_report = metrics.get("classification_report", {})
                if clf_report:
                    report = dict(clf_report)
                    # sklearn stores overall accuracy as a scalar beside the
                    # per-class dicts; transposing it blindly broadcasts that
                    # value into every column, so lift it out first.
                    overall_accuracy = report.pop("accuracy", None)

                    df_rep = pd.DataFrame(report).transpose()
                    df_rep.index = [
                        {
                            "macro avg": "Macro average",
                            "weighted avg": "Weighted average",
                        }.get(str(index), str(index))
                        for index in df_rep.index
                    ]
                    df_rep = df_rep.rename(
                        columns={
                            "precision": "Precision",
                            "recall": "Recall",
                            "f1-score": "F1-score",
                            "support": "Support",
                        }
                    )
                    df_rep.index.name = "Class / average"

                    averages = [
                        index
                        for index in df_rep.index
                        if index in ("Macro average", "Weighted average")
                    ]
                    classes = [index for index in df_rep.index if index not in averages]

                    pieces = [df_rep.loc[classes]]
                    if isinstance(overall_accuracy, (int, float)) and not isinstance(overall_accuracy, bool):
                        total_support = pd.to_numeric(
                            df_rep.loc[classes, "Support"], errors="coerce"
                        ).sum()
                        pieces.append(
                            pd.DataFrame(
                                [
                                    {
                                        "Precision": np.nan,
                                        "Recall": np.nan,
                                        "F1-score": float(overall_accuracy),
                                        "Support": total_support,
                                    }
                                ],
                                index=pd.Index(["Accuracy"], name="Class / average"),
                            )
                        )
                    if averages:
                        pieces.append(df_rep.loc[averages])
                    df_rep = pd.concat(pieces)

                    st.dataframe(
                        df_rep.style.format(
                            {
                                "Precision": "{:.4f}",
                                "Recall": "{:.4f}",
                                "F1-score": "{:.4f}",
                                "Support": "{:,.0f}",
                            },
                            na_rep="—",
                        ),
                        width="stretch",
                    )
                else:
                    st.info("Classification report not recorded for this model.")
            else:
                metric_rows = [
                    ("R² score", metrics.get("r2"), "Test"),
                    ("Explained variance", metrics.get("explained_variance"), "Test"),
                    ("RMSE", metrics.get("rmse"), "Test"),
                    ("MAE", metrics.get("mae"), "Test"),
                    ("MSE", metrics.get("mse"), "Test"),
                    ("Median absolute error", metrics.get("median_ae"), "Test"),
                    ("Max error", metrics.get("max_error"), "Test"),
                    ("MAPE", metrics.get("mape"), "Test"),
                    ("Train R²", metrics.get("train_r2"), "Train"),
                    ("Train RMSE", metrics.get("train_rmse"), "Train"),
                    ("Overfit gap (R²)", metrics.get("overfit_gap"), "Diagnostic"),
                    ("Residual mean", metrics.get("residual_mean"), "Diagnostic"),
                    ("Residual std", metrics.get("residual_std"), "Diagnostic"),
                ]
                df_metrics = pd.DataFrame(
                    [
                        {"Metric": name, "Value": float(value), "Split": split}
                        for name, value, split in metric_rows
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    ]
                )
                if not df_metrics.empty:
                    st.dataframe(
                        df_metrics.style.format({"Value": _format_metric}),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("No scalar metrics recorded for this model.")

            with st.expander("Raw metric payload", expanded=False):
                st.json(metrics)

            # Model artifact download
            artifact_path = metrics.get("artifact_path")
            if artifact_path and Path(artifact_path).exists():
                st.markdown(
                    ui.section("Export trained model", "Serialized pipeline including preprocessing."),
                    unsafe_allow_html=True,
                )
                with open(artifact_path, "rb") as f:
                    joblib_bytes = f.read()
                st.download_button(
                    label=f"Download {selected_model['model_name']} (.joblib)",
                    data=joblib_bytes,
                    file_name=Path(artifact_path).name,
                    mime="application/octet-stream",
                    help="Serialized model and preprocessor pipeline.",
                )

    st.markdown(ui.footer(), unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 4: Live Inference (What-If Tool)
# ------------------------------------------------------------------------------
elif menu == "⚡ Live Inference":
    st.markdown(
        ui.hero(
            "Live Inference",
            "Run single-sample predictions through any model trained in this session and inspect "
            "class probabilities in real time.",
            meta=["In-memory pipelines", "What-if inputs", "Class probabilities", "Instant prediction"],
            eyebrow=f"benchmark console · {THEME} mode",
        ),
        unsafe_allow_html=True,
    )

    pipeline: Optional[TrainingPipeline] = st.session_state.active_pipeline

    if pipeline is None or not pipeline.fitted_trainers:
        st.markdown(
            ui.callout(
                "<strong>No active pipeline.</strong> Train models in "
                "<em>Train New Models</em> first — the fitted pipeline stays in memory for this session.",
                "warning",
            ),
            unsafe_allow_html=True,
        )
    else:
        total_trainers = sum(len(rows) for rows in pipeline.fitted_trainers.values())
        st.markdown(
            ui.callout(
                f"<strong>Active pipeline loaded</strong> · target <code>{pipeline.target_col}</code> "
                f"({pipeline.task_type.capitalize()}) · "
                f"<code>{total_trainers}</code> trained models across "
                f"<code>{len(pipeline.fitted_trainers)}</code> variant(s).",
                "success",
            ),
            unsafe_allow_html=True,
        )

        variant_names = list(pipeline.fitted_trainers.keys())
        if len(variant_names) > 1:
            active_variant = st.selectbox(
                "Data variant",
                options=variant_names,
                format_func=_variant_label,
                key="inference_variant",
            )
        else:
            active_variant = variant_names[0]
            st.caption(f"Variant: **{_variant_label(active_variant)}**")

        available_trainers = pipeline.fitted_trainers[active_variant]
        chosen_mid = st.selectbox(
            "Select Trained Model for Inference",
            options=list(available_trainers.keys()),
            format_func=lambda x: MODEL_REGISTRY[x].display_name,
        )
        trainer = available_trainers[chosen_mid]
        variant_data = pipeline.variants[active_variant]

        st.markdown(
            ui.section("Input feature values", "Ranges and options are derived from the training split."),
            unsafe_allow_html=True,
        )
        # Build form dynamically from X_train_raw
        raw_cols = pipeline.X_train_raw.columns.tolist()
        num_cols = variant_data.preprocessor.num_cols
        cat_cols = variant_data.preprocessor.cat_cols

        input_data = {}
        cols_grid = st.columns(3)

        for i, col in enumerate(raw_cols):
            with cols_grid[i % 3]:
                if col in num_cols:
                    mean_val = float(pipeline.X_train_raw[col].mean())
                    min_val = float(pipeline.X_train_raw[col].min())
                    max_val = float(pipeline.X_train_raw[col].max())
                    input_data[col] = st.number_input(
                        f"{col}",
                        value=mean_val,
                        min_value=min_val,
                        max_value=max_val,
                        key=f"input_{col}",
                    )
                else:
                    unique_vals = pipeline.X_train_raw[col].dropna().unique().tolist()
                    input_data[col] = st.selectbox(
                        f"{col}",
                        options=unique_vals if unique_vals else ["Unknown"],
                        key=f"input_{col}",
                    )

        if st.button("⚡ Run Prediction", type="primary", width="stretch"):
            try:
                # Apply the variant's fitted treatment (clipping / smoothing)
                # before the preprocessor, so live inputs match training data.
                sample = pd.DataFrame([input_data])
                if variant_data.noise_treater is not None:
                    sample = variant_data.noise_treater.transform(sample)
                if variant_data.outlier_treater is not None:
                    sample, _ = variant_data.outlier_treater.transform(sample)
                X_proc, _ = variant_data.preprocessor.transform(
                    sample, task=pipeline.task_type
                )
                raw_pred = trainer.predict(X_proc)

                st.markdown(
                    ui.section("Prediction results", "Live output from the selected model."),
                    unsafe_allow_html=True,
                )
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    if pipeline.task_type == TASK_CLASSIFICATION:
                        label = variant_data.preprocessor.inverse_transform_target(raw_pred)[0]
                        st.markdown(
                            ui.kpi(
                                "Predicted class",
                                str(label),
                                f"Model: {trainer.display_name}",
                                tone="warn",
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        val = float(raw_pred[0])
                        st.markdown(
                            ui.kpi(
                                "Predicted value",
                                f"{val:.4f}",
                                f"Target: {pipeline.target_col}",
                                tone="warn",
                            ),
                            unsafe_allow_html=True,
                        )

                with res_col2:
                    if pipeline.task_type == TASK_CLASSIFICATION and hasattr(trainer, "predict_proba"):
                        proba = trainer.predict_proba(X_proc)
                        if proba is not None:
                            classes = (
                                variant_data.preprocessor.classes_
                                if variant_data.preprocessor.classes_ is not None
                                else range(proba.shape[1])
                            )
                            df_probs = pd.DataFrame(
                                {"Class": [str(c) for c in classes], "Probability": proba[0]}
                            )
                            st.dataframe(
                                df_probs.style.format({"Probability": "{:.2%}"}),
                                width="stretch",
                                hide_index=True,
                            )
            except Exception as e:
                st.error(f"Inference error: {e}")

    st.markdown(ui.footer(), unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 5: Noise & Data Quality
# ------------------------------------------------------------------------------
elif menu == "🧪 Noise & Data Quality":
    st.markdown(
        ui.hero(
            "Noise & Data Quality",
            "Inspect missingness, outliers and high-frequency noise, then compare the same split "
            "trained with and without treatment.",
            meta=["Per-feature profile", "Baseline vs treated", "Outlier audit", "Split provenance"],
            eyebrow=f"benchmark console · {THEME} mode",
        ),
        unsafe_allow_html=True,
    )

    all_experiments = get_all_experiments()
    if not all_experiments:
        st.info("No experiments yet. Train models first to profile data quality.")
    else:
        exp_id = st.session_state.selected_exp_id or all_experiments[0]["id"]
        exp_info = get_experiment(exp_id)
        data_quality = _json_field(exp_info, "data_quality_json")
        variants = get_variants(exp_id)
        task_type = exp_info["task_type"]

        if not data_quality:
            st.markdown(
                ui.callout(
                    "<strong>No data-quality profile.</strong> This experiment predates the "
                    "profiling feature. Run a new training to populate missingness, outlier and "
                    "noise statistics.",
                    "warning",
                ),
                unsafe_allow_html=True,
            )
        else:
            summary = _quality_summary(data_quality)
            columns_profile = summary["columns"]
            outlier_values = [
                stats.get("outlier_iqr_pct")
                for stats in columns_profile.values()
                if isinstance(stats.get("outlier_iqr_pct"), (int, float))
            ]
            mean_outlier = float(np.mean(outlier_values)) if outlier_values else 0.0
            target_info = summary["target"]

            st.markdown(
                ui.kpi_grid(
                    [
                        {
                            "label": "Rows profiled",
                            "value": f"{data_quality.get('n_rows', 0):,}",
                            "hint": f"{data_quality.get('n_cols', 0)} columns · {data_quality.get('n_numeric', 0)} numeric",
                            "tone": "neutral",
                        },
                        {
                            "label": "Missing cells",
                            "value": f"{data_quality.get('missing_pct', 0):.2f}%",
                            "hint": f"{data_quality.get('missing_cells', 0):,} cells",
                            "tone": "info",
                        },
                        {
                            "label": "Mean IQR outliers",
                            "value": f"{mean_outlier:.2f}%",
                            "hint": "Per numeric feature",
                            "tone": "warn",
                        },
                        {
                            "label": "Target health",
                            "value": str(target_info.get("severity") or "—"),
                            "hint": (
                                f"{target_info.get('n_classes', '—')} classes · ratio "
                                f"{target_info.get('imbalance_ratio', '—')}"
                                if task_type == TASK_CLASSIFICATION
                                else "Continuous target"
                            ),
                            "tone": "good",
                        },
                    ],
                    columns=4,
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                ui.section(
                    "Treatment configuration",
                    "What was applied to the treated variant of this experiment.",
                ),
                unsafe_allow_html=True,
            )
            noise_cfg = summary["noise"]
            outlier_cfg = summary["outliers"]
            injection_cfg = summary["injection"]
            st.markdown(
                ui.callout(
                    f"<strong>Denoising</strong> · <code>{noise_cfg.get('strategy', 'none')}</code>"
                    f" · window <code>{noise_cfg.get('window', '—')}</code>"
                    f" · columns <code>{len(noise_cfg.get('columns') or [])}</code><br>"
                    f"<strong>Outliers</strong> · <code>{outlier_cfg.get('method', 'none')}</code>"
                    f" / <code>{outlier_cfg.get('action', '—')}</code>"
                    f" · removed <code>{outlier_cfg.get('removed_rows', 0)}</code> training rows<br>"
                    f"<strong>Injection</strong> · <code>{injection_cfg.get('method', 'none')}</code>"
                    + (
                        f" · {injection_cfg.get('cells_changed', 0):,} cells changed"
                        if injection_cfg.get("enabled")
                        else ""
                    ),
                    "info",
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                ui.section(
                    "Noise profile",
                    "High-frequency energy before and after treatment, per feature.",
                ),
                unsafe_allow_html=True,
            )
            with st.container(key="frame3d-noise-lab-profile"):
                st.plotly_chart(
                    charts.plot_noise_profile(data_quality),
                    config=charts.PLOTLY_CONFIG,
                    width="stretch",
                )
            st.markdown(
                ui.chart_description(
                    "The ratio is <code>std(x − rolling_median(x, 5)) / std(x)</code>. A large "
                    "drop after treatment means smoothing removed real high-frequency wobble; "
                    "an unchanged ratio means the filter was too weak for that feature."
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                ui.section(
                    "Per-feature audit",
                    "Missingness, outlier rates and noise energy for one variant.",
                ),
                unsafe_allow_html=True,
            )
            audit_variant = (
                st.selectbox(
                    "Variant to audit",
                    options=variants or [VARIANT_BASELINE],
                    format_func=_variant_label,
                    key="quality_audit_variant",
                )
                if variants
                else VARIANT_BASELINE
            )
            audit_df = _quality_columns_df(data_quality, audit_variant)
            if not audit_df.empty:
                st.dataframe(
                    audit_df.style.format(
                        {
                            "Missing %": "{:.2f}",
                            "IQR outliers %": "{:.2f}",
                            "Z outliers %": "{:.2f}",
                            "Skew": "{:.2f}",
                            "HF noise ratio": "{:.3f}",
                            "Roughness": "{:.3f}",
                        },
                        na_rep="—",
                    ),
                    width="stretch",
                    hide_index=True,
                )

            # ---- Baseline vs treated ablation ---------------------------
            baseline_lb = (
                get_leaderboard(exp_id, variant=VARIANT_BASELINE)
                if len(variants) > 1
                else []
            )
            treated_lb = (
                get_leaderboard(exp_id, variant=VARIANT_TREATED)
                if VARIANT_TREATED in variants
                else []
            )
            primary_metric = primary_metric_for(task_type)
            primary_label = primary_label_for(task_type)

            if baseline_lb and treated_lb:
                comparison = compare_variants(
                    baseline_lb, treated_lb, primary_metric, higher_is_better=True
                )
                st.markdown(
                    ui.section(
                        "Noisy vs treated",
                        "Paired results on the identical split and test rows.",
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    ui.kpi_grid(
                        [
                            {
                                "label": f"Mean Δ {primary_label}",
                                "value": f"{comparison['mean_delta']:+.4f}",
                                "hint": "Treated minus baseline",
                                "tone": "good" if comparison["mean_delta"] > 0 else "danger",
                            },
                            {
                                "label": "Models improved",
                                "value": str(comparison["wins"]),
                                "hint": f"{comparison['losses']} degraded · {comparison['unchanged']} unchanged",
                                "tone": "info",
                            },
                            {
                                "label": "Best baseline",
                                "value": _shorten(comparison["baseline_best"].get("model") or "—", 26),
                                "hint": f"{comparison['baseline_best'].get('score')}",
                                "tone": "neutral",
                            },
                            {
                                "label": "Best treated",
                                "value": _shorten(comparison["treated_best"].get("model") or "—", 26),
                                "hint": f"{comparison['treated_best'].get('score')}",
                                "tone": "warn",
                            },
                        ],
                        columns=4,
                    ),
                    unsafe_allow_html=True,
                )
                noise_col1, noise_col2 = st.columns(2)
                with noise_col1:
                    with st.container(key="frame3d-noise-variant"):
                        st.plotly_chart(
                            charts.plot_variant_comparison(
                                baseline_lb,
                                treated_lb,
                                primary_metric,
                                primary_label,
                                higher_is_better=True,
                            ),
                            config=charts.PLOTLY_CONFIG,
                            width="stretch",
                        )
                with noise_col2:
                    with st.container(key="frame3d-noise-delta"):
                        st.plotly_chart(
                            charts.plot_variant_delta(
                                baseline_lb,
                                treated_lb,
                                primary_metric,
                                primary_label,
                                higher_is_better=True,
                            ),
                            config=charts.PLOTLY_CONFIG,
                            width="stretch",
                        )
                st.markdown(
                    ui.chart_description(
                        "Both variants share the same rows and test set, so differences are "
                        "caused only by the treatment step. Delta bars are oriented so right "
                        "always means the treated data scored better."
                    ),
                    unsafe_allow_html=True,
                )
                with st.expander("Per-model ablation table", expanded=False):
                    ablation_df = pd.DataFrame(
                        [
                            {
                                "Model": row["model_name"],
                                "Baseline": row["baseline"],
                                "Treated": row["treated"],
                                "Δ": row["delta"],
                                "% change": row["pct_change"],
                                "Verdict": row["direction"],
                                "Rank (baseline)": row["rank_baseline"],
                                "Rank (treated)": row["rank_treated"],
                            }
                            for row in comparison["models"]
                        ]
                    )
                    st.dataframe(
                        ablation_df.style.format(
                            {
                                "Baseline": "{:.4f}",
                                "Treated": "{:.4f}",
                                "Δ": "{:+.4f}",
                                "% change": "{:+.2f}",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                # ---- Distribution shift for one column ------------------
                histograms = (
                    ((data_quality.get("variants") or {}).get(VARIANT_TREATED) or {}).get(
                        "histograms"
                    )
                    or {}
                )
                if histograms:
                    st.markdown(
                        ui.section(
                            "Distribution shift",
                            "Before / after histograms for the treated columns.",
                        ),
                        unsafe_allow_html=True,
                    )
                    hist_col = st.selectbox(
                        "Feature", options=list(histograms.keys()), key="quality_hist_col"
                    )
                    with st.container(key="frame3d-noise-distribution"):
                        st.plotly_chart(
                            charts.plot_column_distribution(
                                histograms.get(hist_col) or {}, hist_col
                            ),
                            config=charts.PLOTLY_CONFIG,
                            width="stretch",
                        )
            else:
                st.markdown(
                    ui.callout(
                        "<strong>No paired comparison.</strong> This run trained a single "
                        "variant. Enable denoising or outlier handling and tick "
                        "<em>Also train the untreated baseline</em> to produce the ablation.",
                        "info",
                    ),
                    unsafe_allow_html=True,
                )

    st.markdown(ui.footer(), unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 6: AI Interpretation
# ------------------------------------------------------------------------------
elif menu == "🤖 AI Interpretation":
    st.markdown(
        ui.hero(
            "AI Interpretation",
            "A grounded report on the active experiment: executive summary, every chart explained, "
            "and a measured verdict on noise and outlier treatment.",
            meta=[f"Groq · {AI_MODEL}", "Chart-aware", "Cached per experiment", "Markdown export"],
            eyebrow=f"benchmark console · {THEME} mode",
        ),
        unsafe_allow_html=True,
    )

    all_experiments = get_all_experiments()
    if not all_experiments:
        st.info("No experiments yet. Train models first, then generate an interpretation.")
    else:
        exp_id = st.session_state.selected_exp_id or all_experiments[0]["id"]
        exp_info = get_experiment(exp_id)
        variants = get_variants(exp_id)
        focus_variant = default_variant(exp_id)
        leaderboard = get_leaderboard(exp_id, variant=focus_variant)
        baseline_lb = (
            get_leaderboard(exp_id, variant=VARIANT_BASELINE) if len(variants) > 1 else []
        )
        treated_lb = (
            get_leaderboard(exp_id, variant=VARIANT_TREATED)
            if VARIANT_TREATED in variants
            else []
        )

        # ---- Provider settings -----------------------------------------
        base_settings = resolve_settings()
        with st.expander("🔐 Provider & credentials", expanded=not base_settings.is_configured):
            provider_options = list(PROVIDERS.keys())
            provider_index = (
                provider_options.index(base_settings.provider)
                if base_settings.provider in provider_options
                else 0
            )
            cfg_col1, cfg_col2 = st.columns(2)
            with cfg_col1:
                provider = st.selectbox(
                    "Provider", options=provider_options, index=provider_index,
                    format_func=lambda key: PROVIDERS[key]["label"],
                )
            preset = PROVIDERS[provider]
            with cfg_col2:
                api_key_override = st.text_input(
                    "API key (session only — never stored)",
                    type="password",
                    value="",
                    help="Leave empty to use GROQ_API_KEY from .env / environment / Streamlit secrets.",
                )
            cfg_col3, cfg_col4 = st.columns(2)
            with cfg_col3:
                base_url = st.text_input(
                    "Base URL",
                    value=(
                        base_settings.base_url
                        if base_settings.provider == provider
                        else preset["base_url"]
                    ),
                    key=f"ai_base_url_{provider}",
                )
            with cfg_col4:
                model = st.text_input(
                    "Model",
                    value=(
                        base_settings.model
                        if base_settings.provider == provider
                        else preset["default_model"]
                    ),
                    key=f"ai_model_{provider}",
                )
            st.caption(
                "Precedence: this form → environment → .env → Streamlit secrets. "
                "Groq keys start with `gsk_`."
            )
            settings = resolve_settings(
                provider=provider,
                api_key=api_key_override or None,
                base_url=base_url,
                model=model,
            )

        st.markdown(
            ui.kpi_grid(
                [
                    {
                        "label": "Provider",
                        "value": PROVIDERS.get(settings.provider, {}).get("label", settings.provider),
                        "hint": settings.host,
                        "tone": "info",
                    },
                    {
                        "label": "Model",
                        "value": _shorten(settings.model, 26),
                        "hint": f"Prompt {AI_PROMPT_VERSION}",
                        "tone": "neutral",
                    },
                    {
                        "label": "API key",
                        "value": settings.masked_key(),
                        "hint": settings.key_source,
                        "tone": "good" if settings.is_configured else "danger",
                    },
                    {
                        "label": "Active experiment",
                        "value": f"#{exp_id}",
                        "hint": _shorten(exp_info["run_name"], 30),
                        "tone": "warn",
                    },
                ],
                columns=4,
            ),
            unsafe_allow_html=True,
        )

        if not settings.is_configured:
            st.markdown(
                ui.callout(
                    "<strong>No API key found.</strong> Create a <code>.env</code> file next to "
                    "<code>config.py</code> containing <code>GROQ_API_KEY=gsk_...</code>, add it "
                    "to <code>.streamlit/secrets.toml</code>, or paste it in the panel above.",
                    "warning",
                ),
                unsafe_allow_html=True,
            )

        # ---- Report controls -------------------------------------------
        model_names = [row["model_name"] for row in leaderboard]
        if not model_names:
            st.warning("This experiment has no model results to interpret.")
        else:
            primary_metric = primary_metric_for(exp_info["task_type"])
            best_row = max(
                leaderboard,
                key=lambda row: (
                    (row.get("metrics") or {}).get(primary_metric)
                    if isinstance((row.get("metrics") or {}).get(primary_metric), (int, float))
                    else -1e18
                ),
            )
            focus_col, action_col = st.columns([1.6, 1])
            with focus_col:
                focus_model = st.selectbox(
                    "Model focus for deep-dive charts",
                    options=model_names,
                    index=model_names.index(best_row["model_name"]),
                )
            with action_col:
                st.write("")
                test_clicked = st.button(
                    "🔌 Test connection", width="stretch", disabled=not settings.is_configured
                )

            if test_clicked:
                with st.spinner("Contacting provider..."):
                    ok, message = test_connection(settings)
                (st.success if ok else st.error)(message)

            cached = _cached_ai_report(exp_id, settings.model)
            button_label = "🤖 Generate full report" if not cached else "♻️ Regenerate full report"
            generate_clicked = st.button(
                button_label,
                type="primary",
                width="stretch",
                disabled=not settings.is_configured,
            )

            if generate_clicked:
                context = build_insight_context(
                    exp_info,
                    leaderboard,
                    baseline_leaderboard=baseline_lb or None,
                    treated_leaderboard=treated_lb or None,
                    focus_model=focus_model,
                )
                try:
                    with st.spinner("Analysing the experiment..."):
                        report, meta = generate_report(context, settings)
                    save_ai_insight(
                        experiment_id=exp_id,
                        prompt_version=AI_PROMPT_VERSION,
                        provider=settings.provider,
                        model=settings.model,
                        context_hash=meta.get("context_hash", ""),
                        content={**report, "_meta": meta},
                    )
                    st.success("Interpretation generated and cached.")
                    st.rerun()
                except AIInsightError as exc:
                    st.error(str(exc))
                except Exception as exc:  # pragma: no cover - defensive
                    st.error(f"Unexpected error: {exc}")

            # ---- Render the cached report -------------------------------
            report_record = _cached_ai_report(exp_id, settings.model)
            if not report_record:
                st.markdown(
                    ui.callout(
                        "No report cached for this experiment and model yet. Press "
                        "<strong>Generate full report</strong> — the result is stored in SQLite "
                        "so switching pages will not call the API again.",
                        "info",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                meta = report_record.get("_meta") or {}
                st.markdown(
                    ui.section(
                        "Executive summary",
                        f"Generated {meta.get('model', settings.model)} · prompt "
                        f"{meta.get('prompt_version', AI_PROMPT_VERSION)} · "
                        f"{meta.get('prompt_chars', '—')} chars of context",
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(report_record.get("summary") or "_No summary returned._")

                findings = report_record.get("key_findings") or []
                if findings:
                    st.markdown(ui.section("Key findings"), unsafe_allow_html=True)
                    for item in findings:
                        st.markdown(f"- {item}")

                analysis = report_record.get("model_analysis") or {}
                if analysis:
                    st.markdown(ui.section("Model analysis"), unsafe_allow_html=True)
                    analysis_cols = st.columns(min(2, max(1, len(analysis))))
                    for index, (key, value) in enumerate(analysis.items()):
                        with analysis_cols[index % len(analysis_cols)]:
                            st.markdown(f"**{str(key).replace('_', ' ').title()}**")
                            st.markdown(str(value))

                explanations = report_record.get("chart_explanations") or {}
                if explanations:
                    st.markdown(
                        ui.section(
                            "Chart-by-chart explanation",
                            "Each entry corresponds to a chart on another page.",
                        ),
                        unsafe_allow_html=True,
                    )
                    for key, text in explanations.items():
                        location = CHART_LOCATIONS.get(key, "")
                        title = key.replace("_", " ").title()
                        label = f"{title}" + (f" · {location}" if location else "")
                        with st.expander(label, expanded=False):
                            st.markdown(str(text))

                impact_sections = [
                    ("noise_impact", "Impact of noise & treatment"),
                    ("outlier_impact", "Impact of outliers"),
                    ("imputation_impact", "Impact of imputation"),
                ]
                for field, title in impact_sections:
                    if report_record.get(field):
                        st.markdown(ui.section(title), unsafe_allow_html=True)
                        st.markdown(str(report_record[field]))

                for field, title in (
                    ("recommendations", "Recommendations"),
                    ("limitations", "Limitations"),
                ):
                    items = report_record.get(field) or []
                    if items:
                        st.markdown(ui.section(title), unsafe_allow_html=True)
                        for item in items:
                            st.markdown(f"- {item}")

                markdown_report = report_to_markdown(report_record, meta)
                st.download_button(
                    "Download report (.md)",
                    data=markdown_report.encode("utf-8"),
                    file_name=f"ai_interpretation_exp_{exp_id}.md",
                    mime="text/markdown",
                    width="stretch",
                )

    st.markdown(ui.footer(), unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 7: Experiment History & DB
# ------------------------------------------------------------------------------
elif menu == "🗄️ Experiment History":
    st.markdown(
        ui.hero(
            "Experiment History",
            "Every benchmark run is persisted in SQLite — reopen, export or prune them here.",
            meta=["SQLite storage", "CSV export", "Run management", "Benchmark archive"],
            eyebrow=f"benchmark console · {THEME} mode",
        ),
        unsafe_allow_html=True,
    )

    all_experiments = get_all_experiments()

    if not all_experiments:
        st.info("No recorded experiments found in SQLite.")
    else:
        latest = all_experiments[0]
        st.markdown(
            ui.kpi_grid(
                [
                    {
                        "label": "Experiments stored",
                        "value": f"{len(all_experiments):,}",
                        "hint": "SQLite · WAL mode",
                        "tone": "info",
                    },
                    {
                        "label": "Latest run",
                        "value": _shorten(latest["run_name"]),
                        "hint": f"#{latest['id']} · {latest['task_type'].capitalize()}",
                        "tone": "neutral",
                    },
                ],
                columns=2,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            ui.section("Benchmark progress", "Best primary score across stored experiments."),
            unsafe_allow_html=True,
        )
        with st.container(key="frame3d-history-trend"):
            st.plotly_chart(
                charts.plot_experiment_trend(get_benchmark_history()),
                config=charts.PLOTLY_CONFIG,
                width="stretch",
            )
        st.markdown(
            ui.chart_description(
                "One point per experiment, split by task type because classification F1 and "
                "regression R² are not directly comparable. Hover a point for the run name, "
                "dataset and winning model."
            ),
            unsafe_allow_html=True,
        )

        df_exps = pd.DataFrame(all_experiments)
        st.dataframe(df_exps, width="stretch", hide_index=True)

        st.markdown(
            ui.section("Maintenance", "Export a leaderboard or remove a stored run."),
            unsafe_allow_html=True,
        )
        h_col1, h_col2 = st.columns(2)

        with h_col1:
            st.markdown(
                ui.chart_header("Export benchmark results", "csv"),
                unsafe_allow_html=True,
            )
            exp_to_export = st.selectbox(
                "Select Experiment to Export",
                options=[e["id"] for e in all_experiments],
                format_func=lambda x: f"Exp #{x}: {next(e['run_name'] for e in all_experiments if e['id'] == x)}",
            )
            df_exp = export_leaderboard_df(exp_to_export)
            if not df_exp.empty:
                csv_data = df_exp.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Leaderboard as CSV",
                    data=csv_data,
                    file_name=f"benchmark_exp_{exp_to_export}.csv",
                    mime="text/csv",
                    width="stretch",
                )

        with h_col2:
            st.markdown(
                ui.chart_header("Delete experiment", "irreversible"),
                unsafe_allow_html=True,
            )
            exp_to_delete = st.selectbox(
                "Select Experiment to Remove",
                options=[e["id"] for e in all_experiments],
                format_func=lambda x: f"Exp #{x}: {next(e['run_name'] for e in all_experiments if e['id'] == x)}",
                key="del_exp_sel",
            )
            if st.button("Delete Experiment", type="secondary", width="stretch"):
                delete_experiment(exp_to_delete)
                st.success(f"Experiment #{exp_to_delete} deleted!")
                st.rerun()

    st.markdown(ui.footer(), unsafe_allow_html=True)

