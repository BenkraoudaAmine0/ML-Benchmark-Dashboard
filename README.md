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
  - Numeric features: Median imputation followed by optional z-score standardization (`StandardScaler`).
  - Categorical features: Most-frequent imputation followed by one-hot encoding (`OneHotEncoder` with unknown category handling).
  - Target variables: Label encoding for categorical targets with automatic inverse transformation during prediction.

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

The web application (`frontend/app.py`) is structured into five focused views:

1. Comparison Dashboard
   - Top KPI metric cards showing the leading model, peak score, lowest latency, and dataset dimensions.
   - Leaderboard table sorted by primary evaluation score (F1-Macro for classification, R-squared for regression).
   - Side-by-side Plotly bar charts for metric comparisons and training time overhead.
   - Multi-metric radar comparison charts.
   - Classical vs. Ensemble aggregate benchmark summaries.
   - Interactive head-to-head model matchup table.

2. Train New Models
   - Dataset selection (or upload) with automated target and task detection.
   - Configuration sliders for test split ratio, random seed, and feature scaling.
   - Row subsampling controls (random sampling or sequential ordering) for managing training times on large datasets.
   - Model selection presets: The 8 Core Models, All 12 Models, Classical Only, or Ensembles Only.
   - Collapsible panel to override key hyperparameters (e.g., SVM kernels, tree depths, learning rates, estimator counts).
   - Live progress indicator with status updates during training.

3. Model Deep Dive
   - Generalization banner highlighting train-test gaps and overfitting risks.
   - Confusion matrix heatmaps for classification tasks.
   - Actual vs. Predicted scatter plots and residual analysis for regression tasks.
   - Top feature importances chart (for tree-based and linear models).
   - Detailed classification report or raw metrics breakdown.
   - Single-click download button for the serialized `.joblib` model artifact.

4. Live Inference (What-If Analysis)
   - Inspects the active in-memory trained pipeline.
   - Dynamically builds an interactive input form derived from the dataset's features, respecting numeric ranges and categorical options.
   - Executes real-time prediction and displays predicted classes with class probability tables (classification) or target values (regression).

5. Experiment History
   - Searchable view of all historical runs stored in the SQLite database.
   - One-click CSV export of full benchmark leaderboards.
   - Run deletion utility for database maintenance.

---

## Project Structure

```
ML-Benchmark-Dashboard/
├── config.py                 # File paths, task constants, and SQLite database settings
├── core/
│   ├── __init__.py
│   ├── base.py               # Abstract BaseModelTrainer interface and contracts
│   ├── data_loader.py        # Dataset discovery, loading, task inference, train/test splitting
│   ├── preprocessor.py       # Scikit-learn ColumnTransformer for scaling and encoding
│   ├── metrics.py            # Classification and regression metric computation suites
│   ├── storage.py            # SQLite database manager (WAL mode) and artifact tracking
│   └── pipeline.py           # Training orchestrator managing runs, metrics, and persistence
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
│   ├── app.py                # Main Streamlit dashboard application
│   ├── charts.py             # Reusable Plotly chart definitions (radar, bars, residuals, CM)
│   └── ui_theme.py           # Custom CSS styling, headers, and KPI card components
├── storage/
│   ├── training_results.db   # SQLite database storing experiments, runs, and metrics
│   └── artifacts/            # Serialized model and pipeline joblib files
├── test_pipeline.py          # Smoke tests on synthetic classification and regression datasets
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
pip install numpy pandas scikit-learn xgboost lightgbm catboost streamlit plotly joblib
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

### 3. Run Benchmarks on Real Datasets
To execute a headless benchmark against real CSV files located in your datasets folder:

```bash
python verify_real_datasets.py
```

---

## Configuration

Default paths and runtime constants are defined in `config.py`:
- `DATASETS_DIR`: Directory scanned for CSV datasets (defaults to `../Datasets`).
- `STORAGE_DIR`: Directory where SQLite files and serialized model artifacts reside (`storage/`).
- `DB_PATH`: Location of the SQLite database (`storage/training_results.db`).
- `ARTIFACTS_DIR`: Directory containing `.joblib` model dumps (`storage/artifacts/`).
