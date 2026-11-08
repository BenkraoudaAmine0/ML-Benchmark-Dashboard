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
    DATASETS_DIR,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
)
from core.data_loader import (
    detect_task_type,
    discover_datasets,
    load_dataset,
)
from core.pipeline import TrainingPipeline
from core.storage import (
    delete_experiment,
    export_leaderboard_df,
    get_all_experiments,
    get_experiment,
    get_leaderboard,
    init_db,
)
from frontend import charts
from frontend import ui_theme as ui
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


# ------------------------------------------------------------------------------
# Sidebar Navigation & Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    ui.sidebar_brand()
    ui.theme_switcher()

    st.markdown(
        ui.section("Navigate", "Five focused workspaces."),
        unsafe_allow_html=True,
    )
    menu = st.radio(
        "Navigation",
        [
            "🏆 Comparison Dashboard",
            "🚀 Train New Models",
            "🔬 Model Deep Dive",
            "⚡ Live Inference",
            "🗄️ Experiment History",
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
        current_id = st.session_state.selected_exp_id or all_experiments[0]["id"]
        if current_id not in exp_options:
            current_id = all_experiments[0]["id"]

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
        leaderboard = get_leaderboard(exp_id)

        if not leaderboard:
            st.warning("No model results found for this experiment.")
        else:
            task_type = exp_info["task_type"]

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

            # Tabular Leaderboard
            df_leaderboard = export_leaderboard_df(exp_id)
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

            with chart_col2:
                fig_time = charts.plot_latency_comparison(leaderboard)
                with st.container(key="frame3d-latency"):
                    st.plotly_chart(fig_time, config=charts.PLOTLY_CONFIG, width="stretch")

            # Radar & category trade-off
            st.markdown(
                ui.section("Multi-metric comparison", "Radar overlay and architecture-level summary."),
                unsafe_allow_html=True,
            )
            radar_col1, radar_col2 = st.columns([1.2, 1])

            with radar_col1:
                fig_radar = charts.plot_radar_metrics(leaderboard, task_type)
                with st.container(key="frame3d-radar"):
                    st.plotly_chart(fig_radar, config=charts.PLOTLY_CONFIG, width="stretch")

            with radar_col2:
                st.markdown(
                    ui.chart_header("Classical vs ensemble", "aggregate"),
                    unsafe_allow_html=True,
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
            st.table(pd.DataFrame(diff_records))

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
            ui.section("03 · Dataset size & preprocessing", "Split ratio, seed and feature scaling."),
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
        sampling_strategy = "Random Sample"
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
                sampling_strategy = st.radio(
                    "Selection Method",
                    ["Random Sample", "First N Rows (By Order)"],
                    horizontal=True,
                    help="Choose 'Random Sample' for uniform shuffling or 'First N Rows' to preserve dataset sequence (e.g. time series).",
                )
                if sampling_strategy == "Random Sample":
                    st.caption(f"⚡ Shuffling randomly using seed `{seed}`.")
                else:
                    st.caption(f"⚡ Taking the first `{max_samples_val:,}` rows in original file order.")

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
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(mid: str, pct: float, msg: str):
                    progress_bar.progress(pct)
                    status_text.markdown(f"⏳ **Status**: {msg}")

                # Downsample if user requested max rows
                if max_samples_val and len(df) > max_samples_val:
                    if sampling_strategy == "First N Rows (By Order)":
                        df_to_use = df.iloc[:max_samples_val].reset_index(drop=True)
                    else:
                        df_to_use = df.sample(n=max_samples_val, random_state=seed).reset_index(drop=True)
                else:
                    df_to_use = df

                # Guard: regression requires a numeric (or numeric-coercible) target
                if task_type == TASK_REGRESSION:
                    raw_target = df_to_use[target_col]
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
                    df=df_to_use,
                    target_col=target_col,
                    task=task_type,
                    test_size=test_size,
                    random_state=seed,
                    scale_numeric=scale_num,
                    max_samples=max_samples_val,
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
                    st.success(f"Benchmark finished! {len(results['results'])} models trained. Head to the **🏆 Comparison Dashboard** to review the results!")
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
        leaderboard = get_leaderboard(exp_id)

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

            # Classification report or residual distribution
            st.markdown(
                ui.section("Performance report", "Per-class precision / recall / support or the full metric payload."),
                unsafe_allow_html=True,
            )
            if task_type == TASK_CLASSIFICATION:
                clf_report = metrics.get("classification_report", {})
                if clf_report:
                    df_rep = pd.DataFrame(clf_report).transpose()
                    st.dataframe(df_rep.style.format(precision=4), width="stretch")
            else:
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
        st.markdown(
            ui.callout(
                f"<strong>Active pipeline loaded</strong> · target <code>{pipeline.target_col}</code> "
                f"({pipeline.task_type.capitalize()}) · "
                f"<code>{len(pipeline.fitted_trainers)}</code> trained models.",
                "success",
            ),
            unsafe_allow_html=True,
        )

        available_trainers = pipeline.fitted_trainers
        chosen_mid = st.selectbox(
            "Select Trained Model for Inference",
            options=list(available_trainers.keys()),
            format_func=lambda x: MODEL_REGISTRY[x].display_name,
        )
        trainer = available_trainers[chosen_mid]

        st.markdown(
            ui.section("Input feature values", "Ranges and options are derived from the training split."),
            unsafe_allow_html=True,
        )
        # Build form dynamically from X_train_raw
        raw_cols = pipeline.X_train_raw.columns.tolist()
        num_cols = pipeline.preprocessor.num_cols
        cat_cols = pipeline.preprocessor.cat_cols

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
                # Preprocess single sample
                X_proc = pipeline.preprocessor.transform_single_dict(input_data)
                raw_pred = trainer.predict(X_proc)

                st.markdown(
                    ui.section("Prediction results", "Live output from the selected model."),
                    unsafe_allow_html=True,
                )
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    if pipeline.task_type == TASK_CLASSIFICATION:
                        label = pipeline.preprocessor.inverse_transform_target(raw_pred)[0]
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
                                pipeline.preprocessor.classes_
                                if pipeline.preprocessor.classes_ is not None
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
# TAB 5: Experiment History & DB
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

