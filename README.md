# ML2 Model Benchmark Studio

A comprehensive, modular machine learning training and benchmarking system supporting both **Classification** and **Regression** tasks.

## 🚀 Features
- **Base Models**: Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Decision Tree
- **Ensemble Models**: Random Forest (Bagging), AdaBoost, XGBoost, LightGBM, CatBoost (Boosting)
- **Modular Architecture**: Each model family has its own dedicated Python module in `models/` adhering to `BaseModelTrainer`.
- **Zero Data Leakage**: Standardized `ColumnTransformer` preprocessing pipeline fitted exclusively on training data.
- **SQLite Database Persistence**: All runs, metrics, hyperparameters, and predictions are stored in `storage/training_results.db`.
- **Interactive Streamlit Dashboard**:
  - Automatically loads the latest **Leaderboard & Comparison Dashboard** upon opening.
  - Multi-metric Plotly bar charts, radar comparisons, latency tradeoffs, and head-to-head model matchups.
  - Interactive training tab with live progress bar.
  - Model Deep-Dive with Confusion Matrix heatmaps, Residual scatter plots, and Feature Importances.
  - Live "What-If" single-sample inference form.
  - Database management and CSV export.

---

## 📂 Architecture

```
Modeling/
├── config.py                 # Paths, DB configuration, task constants
├── core/
│   ├── base.py               # Abstract BaseModelTrainer interface
│   ├── data_loader.py        # Dataset discovery, loading, auto-task detection, train/test split
│   ├── preprocessor.py       # ColumnTransformer for numeric scaling and one-hot encoding
│   ├── metrics.py            # Evaluation metrics suite (Acc, F1, ROC-AUC, R2, RMSE, MAE, etc.)
│   ├── storage.py            # SQLite database persistence layer (WAL mode)
│   └── pipeline.py           # Training pipeline orchestrator
├── models/
│   ├── train_svm.py          # Support Vector Machine (SVC & SVR)
│   ├── train_knn.py          # K-Nearest Neighbors (Classifier & Regressor)
│   ├── train_dtree.py        # Decision Tree (Classifier & Regressor)
│   ├── train_rf.py           # Random Forest (Classifier & Regressor)
│   ├── train_adaboost.py     # AdaBoost (Classifier & Regressor)
│   ├── train_xgboost.py      # XGBoost (Classifier & Regressor)
│   ├── train_lightgbm.py     # LightGBM (Classifier & Regressor)
│   ├── train_catboost.py     # CatBoost (Classifier & Regressor)
│   └── registry.py           # Model discovery & registry
├── frontend/
│   ├── app.py                # Streamlit web application
│   ├── ui_theme.py           # Visual identity, KPI cards, and custom CSS
│   └── charts.py             # Plotly visualizations
├── storage/
│   └── training_results.db   # SQLite database
├── test_pipeline.py          # Smoke tests on synthetic datasets
└── verify_real_datasets.py   # Benchmark on real repository datasets
```

---

## 🏃‍♂️ How to Run

### 1. Launch the Frontend Dashboard
```powershell
.venv\Scripts\streamlit run Modeling/frontend/app.py
```

### 2. Run Pipeline Tests (CLI)
```powershell
.venv\Scripts\python Modeling/test_pipeline.py
.venv\Scripts\python Modeling/verify_real_datasets.py
```

