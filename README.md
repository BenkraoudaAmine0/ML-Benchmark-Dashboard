# ML2 Model Benchmark Studio

A modular machine learning benchmarking framework and interactive dashboard designed to evaluate, compare, and deploy classical and ensemble learning algorithms for classification and regression tasks.

---

## Overview

ML2 Model Benchmark Studio provides an end-to-end environment for training and comparing machine learning models under identical, reproducible conditions. It covers dataset ingestion, automatic task identification, leak-free preprocessing, multi-model execution, SQLite-backed experiment tracking, and an interactive Streamlit frontend for diagnostics and live inference.

The project implements twelve model architectures split across baseline classical algorithms and modern boosting, bagging, and meta-ensemble techniques.

---

## Core Capabilities

### Dual Task Support with Automatic Detection
- Supports both classification (binary and multiclass) and regression problems.
- Automatically infers the task type from the target column based on data types, unique class counts, and heuristics, with manual override capability in the UI.
- Scans and indexes local datasets dynamically from configured directories.

### Leak-Free Data Preprocessing
- Preprocessing pipelines strictly follow scikit-learn's `ColumnTransformer` architecture.
- Transformers are fit exclusively on the training split, completely preventing data leakage into validation/test splits.
- Handles mixed feature types:
  - Numeric features: pluggable imputation (median, mean, mode, **KNN**, iterative MICE, MissForest-style random-forest MICE, constant + missing indicator, linear interpolation, forward fill, or none for NaN-native boosters) followed by optional z-score standardization (`StandardScaler`).
  - Categorical features: most-frequent or constant-category imputation followed by one-hot encoding (`OneHotEncoder` with unknown category handling).
  - Target variables: Label encoding for categorical targets with automatic inverse transformation during prediction.
- Every imputation strategy is fitted inside the training pipeline, so test rows and live-inference samples use the training statistics only.

### Noise, Outlier and Data-Quality Treatment
- Optional denoising of numeric features: rolling mean, rolling median, exponential moving average, Savitzky–Golay, Hampel filter, linear interpolation, forward fill, IQR winsorization and z-score clipping.
- Optional outlier handling: IQR fence, z-score fence or Isolation Forest, with either clipping or removal. Removal only ever touches training rows — the test set stays identical so comparisons remain fair.
- Optional synthetic noise injection (Gaussian, spikes, random missing cells) so a clean dataset still gets a controlled "noisy" side.
- **Baseline vs treated ablation:** when treatment is enabled, the same split and the same models are trained twice (`baseline` and `treated` variants) inside one experiment, and the noise data-quality lab reports the paired delta, wins/losses and rank changes.

### Reproducible Persistent Splits
- Train/test partitions are fingerprinted from dataset content plus split settings (target, test ratio, seed, row limit, sampling strategy, dropped columns).
- Partitions are saved as compressed joblib files in `storage/splits/` and reused on identical runs, so re-training the same configuration never silently re-draws a split.
- Every experiment records its `split_id`; the UI can pin an older saved split explicitly.

### AI Interpretation (Groq)
- A new AI page generates a grounded report for the active experiment: executive summary, key findings, **an explanation of every dashboard chart** (fed the exact plotted values, not screenshots), model-behaviour analysis, and measured verdicts on noise, outlier and imputation impact.
- Uses the Groq Cloud OpenAI-compatible endpoint with `llama-3.3-70b-versatile` by default; xAI Grok, local Ollama and custom endpoints are selectable.
- Reports are cached in SQLite per experiment/model/prompt version, so switching pages or re-rendering never triggers another API call. Regeneration is explicit.

### Overfitting and Generalization Diagnostics
- Tracks performance across both training and test splits.
- Computes the train-to-test performance delta to flag models that have memorized noise versus those that generalize well to unseen data.

### Experiment Tracking and Model Artifacts
- Embedded SQLite storage operating in Write-Ahead Logging (WAL) mode for low-overhead concurrent reads and writes.
- Automatically records dataset metadata, train/test split sizes, execution timestamps, wall-clock training times, hyperparameters, and full evaluation payloads.
- Persists serialized models (`.joblib`) including their preprocessing transformations into a dedicated artifact directory, allowing trained pipelines to be downloaded or reused.

---

## Supported Models

The framework includes twelve models organized into classical baselines, standard ensemble methods, and meta-estimators:

| Category | Model Key | Name | Supported Tasks | Underlying Implementation |
|---|---|---|---|---|
| Classical | `linear` | Linear / Logistic Regression | Classification, Regression | `LogisticRegression` / `Ridge` |
| Classical | `svm` | Support Vector Machine | Classification, Regression | `SVC` (probability=True) / `SVR` |
| Classical | `knn` | K-Nearest Neighbors | Classification, Regression | `KNeighborsClassifier` / `KNeighborsRegressor` |
| Classical | `dtree` | Decision Tree | Classification, Regression | `DecisionTreeClassifier` / `DecisionTreeRegressor` |
| Ensemble | `rf` | Random Forest | Classification, Regression | `RandomForestClassifier` / `RandomForestRegressor` |
| Ensemble | `adaboost` | AdaBoost | Classification, Regression | `AdaBoostClassifier` / `AdaBoostRegressor` |
| Ensemble | `xgboost` | XGBoost | Classification, Regression | `XGBClassifier` / `XGBRegressor` |
| Ensemble | `lightgbm` | LightGBM | Classification, Regression | `LGBMClassifier` / `LGBMRegressor` |
| Ensemble | `catboost` | CatBoost | Classification, Regression | `CatBoostClassifier` / `CatBoostRegressor` |
| Ensemble | `bagging` | Bagging Ensemble | Classification, Regression | `BaggingClassifier` / `BaggingRegressor` |
| Ensemble | `stacking` | Stacking Regressor / Classifier | Classification, Regression | `StackingClassifier` / `StackingRegressor` |
| Ensemble | `voting` | Voting Ensemble | Classification, Regression | `VotingClassifier` (soft) / `VotingRegressor` |

All models follow a unified `BaseModelTrainer` abstract base class, exposing consistent interfaces for initialization, fitting, prediction, probability calculation, and feature importance extraction.

---

## Evaluation Metrics

Every model run is scored against a full evaluation suite tailored to the active task.

### Classification Metrics
- Accuracy and Balanced Accuracy
- F1-Score (Macro and Weighted averages)
- Precision and Recall (Macro averages)
- ROC-AUC (Binary and Multiclass One-vs-Rest via predicted probabilities)
- Log Loss (Cross-Entropy)
- Full Confusion Matrix
- Per-class classification report (precision, recall, f1, support)

### Regression Metrics
- Coefficient of Determination (R-squared)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Median Absolute Error (MedAE)
- Max Error
- Mean Absolute Percentage Error (MAPE)
- Explained Variance Score
- Residual distribution statistics (mean, standard deviation, and sample vectors for plotting)

---

## Streamlit Application Layout

The web application (`frontend/app.py`) is structured into seven focused views:

1. Comparison Dashboard
   - Top KPI metric cards showing the leading model, peak score, lowest latency, and dataset dimensions.
   - Leaderboard table sorted by primary evaluation score (F1-Macro for classification, R-squared for regression).
   - Side-by-side Plotly bar charts for metric comparisons and training time overhead.
   - Multi-metric radar comparison charts, capped to an adjustable top-N selection (default 5).
   - Classical vs. Ensemble aggregate chart: mean score bars with every individual model plotted as a dot.
   - Score-vs-training-time scatter with the non-dominated Pareto frontier.
   - Benchmark matrix heatmap ranking every model per metric, with lower-is-better metrics ranked in the correct direction.
   - Head-to-head relative-advantage chart (with exact values available in a table expander).

2. Train New Models
   - Dataset selection (or upload) with automated target and task detection.
   - Configuration sliders for test split ratio, random seed, and feature scaling.
   - Row subsampling controls (random sampling or sequential ordering) for managing training times on large datasets.
   - Imputation panel: numeric and categorical strategies with their parameters (KNN neighbours/weights, MICE iterations, MissForest estimators, constant fill, missing indicator).
   - Noise-treatment panel: denoising strategy and window, columns, optional row-order column, outlier method/action/threshold, optional synthetic injection, and the baseline-comparison toggle.
   - Split-persistence panel: automatic fingerprint matching, explicit saved-split selection, and a live hint showing whether the next run will reuse split #N or create one.
   - Model selection presets: The 8 Core Models, All 12 Models, Classical Only, or Ensembles Only.
   - Collapsible panel to override key hyperparameters (e.g., SVM kernels, tree depths, learning rates, estimator counts).
   - Live progress indicator with status updates during training.

3. Noise & Data Quality
   - Dataset-level KPIs: rows, missing cells, mean IQR outlier rate, target health (class imbalance severity or continuous-target statistics).
   - Treatment configuration summary (strategy, window, columns, outlier action, removed rows, injection details).
   - Per-feature noise profile: high-frequency energy before and after treatment.
   - Per-variant audit table: missingness, IQR/z-score outlier percentages, skew, noise ratio and roughness.
   - Noisy-vs-treated ablation: mean delta, wins/losses, best model per variant, grouped bars, paired delta chart, per-model table with rank changes.
   - Before/after distribution histograms for treated features.

4. Model Deep Dive
   - Generalization banner highlighting train-test gaps and overfitting risks.
   - Confusion matrix heatmaps for classification tasks.
   - Actual vs. Predicted scatter plots and residual analysis for regression tasks.
   - Top feature importances chart (for tree-based and linear models).
   - Formatted performance report table: per-class precision / recall / F1 / support plus overall accuracy and averages for classification, or the full test/train/diagnostic metric suite for regression. The raw metric payload stays available in a collapsible panel.
   - Single-click download button for the serialized `.joblib` model artifact.

5. Live Inference (What-If Analysis)
   - Inspects the active in-memory trained pipeline.
   - Dynamically builds an interactive input form derived from the dataset's features, respecting numeric ranges and categorical options.
   - Executes real-time prediction and displays predicted classes with class probability tables (classification) or target values (regression).

6. Experiment History
   - Searchable view of all historical runs stored in the SQLite database.
   - Benchmark progress trend chart plotting the best primary score of each stored experiment, split by task type (comparison runs contribute their baseline variant).
   - One-click CSV export of full benchmark leaderboards, with a variant column when the run trained both.
   - Run deletion utility for database maintenance.

7. AI Interpretation
   - Provider/credential panel with presets for Groq Cloud, xAI Grok, local Ollama and custom OpenAI-compatible endpoints; keys are read from the form, environment, `.env` or Streamlit secrets and never stored.
   - Connection test, one-click full-report generation and explicit regeneration.
   - Cached report rendering: executive summary, key findings, model analysis (best model, overfitting, trade-offs, weak models), chart-by-chart explanations with their page locations, noise/outlier/imputation impact, recommendations and limitations.
   - Markdown export of the full report.
   - Optional "AI read" captions under the comparison charts, rendered from the cache with no extra API calls.

---

## Project Structure

```
ML-Benchmark-Dashboard/
├── config.py                 # File paths, task constants, and SQLite database settings
├── core/
│   ├── __init__.py
│   ├── base.py               # Abstract BaseModelTrainer interface and contracts
│   ├── data_loader.py        # Dataset discovery, loading and task inference
│   ├── splits.py             # Fingerprinted, persisted train/test partitions
│   ├── preprocessor.py       # ColumnTransformer with pluggable imputation and scaling
│   ├── noise.py              # Denoising, outlier handling and synthetic noise injection
│   ├── data_quality.py       # Missingness/outlier/noise profiling and variant comparison
│   ├── metrics.py            # Classification and regression metric computation suites
│   ├── ai_insights.py        # Groq / OpenAI-compatible interpretation client and prompts
│   ├── storage.py            # SQLite manager, migrations, splits and AI cache
│   └── pipeline.py           # Training orchestrator with baseline/treated variants
├── models/
│   ├── __init__.py
│   ├── registry.py           # Central registry and factory functions for model trainers
│   ├── train_linear.py       # Linear and Logistic Regression trainers
│   ├── train_svm.py          # Support Vector Machine (SVC and SVR)
│   ├── train_knn.py          # K-Nearest Neighbors classifier and regressor
│   ├── train_dtree.py        # Decision Tree classifier and regressor
│   ├── train_rf.py           # Random Forest classifier and regressor
│   ├── train_adaboost.py     # AdaBoost classifier and regressor
│   ├── train_xgboost.py      # XGBoost classifier and regressor
│   ├── train_lightgbm.py     # LightGBM classifier and regressor
│   ├── train_catboost.py     # CatBoost classifier and regressor
│   ├── train_bagging.py      # Bagging ensemble trainer
│   ├── train_stacking.py     # Stacking meta-ensemble trainer
│   └── train_voting.py       # Voting ensemble trainer
├── frontend/
│   ├── __init__.py
│   ├── app.py                # Main Streamlit dashboard application (7 pages)
│   ├── charts.py             # Reusable Plotly chart definitions (radar, bars, residuals, CM)
│   ├── insight_context.py    # Chart-by-chart data payloads for the AI interpretation
│   └── ui_theme.py           # Custom CSS styling, headers, and KPI card components
├── storage/
│   ├── training_results.db   # SQLite database (experiments, results, splits, AI cache)
│   ├── artifacts/            # Serialized model and pipeline joblib files
│   └── splits/               # Persisted train/test partitions (fingerprinted joblib)
├── test_pipeline.py          # Smoke tests on synthetic classification and regression datasets
├── test_features.py          # Tests for imputers, noise treatment, splits and AI context
└── verify_real_datasets.py   # Benchmark script testing real datasets from the repository
```

---

## Installation and Setup

### Prerequisites
- Python 3.9 or newer (tested with Python 3.10 to 3.14)
- A clean virtual environment is recommended

### Dependencies
Install the required packages:

```bash
pip install numpy pandas scikit-learn scipy xgboost lightgbm catboost streamlit plotly joblib requests
```

`python-dotenv` is optional but recommended if you keep the AI key in a `.env` file:

```bash
pip install python-dotenv
```

---

## Running the Project

### 1. Launch the Streamlit Dashboard
Run the following command from the workspace or repository root:

```bash
streamlit run frontend/app.py
```

The app adds its own project root to `sys.path`, so the checkout folder can have any name and you can also launch it from the workspace root:

```bash
streamlit run ML-Benchmark-Dashboard/frontend/app.py
```

Datasets are discovered from the `Datasets/` (or `dataset/`) folder, looking in the project root first and then in its parent folders.

### 2. Run Smoke Tests (Synthetic Data)
To verify that all 12 models and the SQLite persistence layer function properly without opening the dashboard:

```bash
python test_pipeline.py
```

This runs both classification and regression routines on synthetic data and prints the resulting experiment IDs and metric summaries.

### 3. Run Feature Tests
To verify imputation strategies, noise treatment, persistent splits, variant comparisons and the AI context builder (no network calls):

```bash
python test_features.py
```

### 4. Run Benchmarks on Real Datasets
To execute a headless benchmark against real CSV files located in your datasets folder:

```bash
python verify_real_datasets.py
```

### 5. Configure the AI interpretation (Groq)
Create a `.env` file next to `config.py` (it is gitignored):

```bash
GROQ_API_KEY=gsk_...
# optional overrides
AI_PROVIDER=groq
AI_BASE_URL=https://api.groq.com/openai/v1
AI_MODEL=llama-3.3-70b-versatile
```

Alternatively add `GROQ_API_KEY` to `.streamlit/secrets.toml`, or paste the key into the **🤖 AI Interpretation** page (session only). The default model is `llama-3.3-70b-versatile`; `AI_BASE_URL`/`AI_MODEL` also allow xAI Grok, a local Ollama server (`http://localhost:11434/v1`) or any OpenAI-compatible endpoint.

---

## Configuration

Default paths and runtime constants are defined in `config.py`:
- `DATASETS_DIR`: Directory scanned for CSV datasets (defaults to `../Datasets`).
- `STORAGE_DIR`: Directory where SQLite files and serialized model artifacts reside (`storage/`).
- `DB_PATH`: Location of the SQLite database (`storage/training_results.db`).
- `ARTIFACTS_DIR`: Directory containing `.joblib` model dumps (`storage/artifacts/`).
- `SPLITS_DIR`: Persisted train/test partitions (`storage/splits/`).
- `NUMERIC_IMPUTERS` / `CATEGORICAL_IMPUTERS`: Supported imputation strategies and their labels.
- `NOISE_STRATEGIES` / `OUTLIER_METHODS` / `INJECTION_METHODS`: Treatment catalogue.
- `AI_PROVIDER`, `AI_BASE_URL`, `AI_MODEL`, `AI_PROMPT_VERSION`: AI defaults.

SQLite schema (migrated in place on startup):
- `experiments`: run metadata plus `split_id`, `comparison_mode`, `noise_config_json`, `imputer_config_json`, `data_quality_json`.
- `model_results`: metrics and predictions with a `variant` column (`baseline` / `treated`).
- `dataset_splits`: fingerprinted split records pointing at joblib partitions.
- `ai_insights`: cached AI reports keyed by experiment, prompt version and model.

Existing databases are upgraded automatically by `core.storage.init_db()` using additive `ALTER TABLE` migrations; no manual migration step is required.
