"""
Configuration and constants for ML2 Training Pipeline.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def _resolve_datasets_dir(start: Path) -> Path:
    """Locate the Datasets folder regardless of the checkout's folder name.

    Walks up from the project root and accepts either ``Datasets`` (the
    documented name) or ``dataset`` (the name used in this workspace), so the
    dashboard finds data from the main checkout as well as from git
    worktrees nested inside it.
    """
    for base in (start, *start.parents):
        for name in ("Datasets", "dataset"):
            candidate = base / name
            if candidate.is_dir():
                return candidate
    return start / "Datasets"


DATASETS_DIR = _resolve_datasets_dir(BASE_DIR)
STORAGE_DIR = BASE_DIR / "storage"
DB_PATH = STORAGE_DIR / "training_results.db"
ARTIFACTS_DIR = STORAGE_DIR / "artifacts"
SPLITS_DIR = STORAGE_DIR / "splits"

# Ensure storage directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Supported tasks
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
SUPPORTED_TASKS = [TASK_CLASSIFICATION, TASK_REGRESSION]

# Model categories
CAT_CLASSICAL = "classical"
CAT_ENSEMBLE = "ensemble"

# Data variants stored per experiment. ``baseline`` is the untreated data (with
# synthetic noise injected if the run requested it); ``treated`` is the same
# split after denoising / outlier handling.
VARIANT_BASELINE = "baseline"
VARIANT_TREATED = "treated"
VARIANT_LABELS = {
    VARIANT_BASELINE: "Noisy / untreated",
    VARIANT_TREATED: "Treated",
}

# ------------------------------------------------------------------------------
# Imputation strategies
# ------------------------------------------------------------------------------
# Implemented numeric strategies. Keys double as the values persisted in the DB.
NUMERIC_IMPUTERS = [
    "median",
    "mean",
    "most_frequent",
    "knn",
    "iterative",
    "iterative_rf",
    "constant",
    "interpolate",
    "ffill",
    "none",
]

NUMERIC_IMPUTER_LABELS = {
    "median": "Median — robust default",
    "mean": "Mean — symmetric distributions",
    "most_frequent": "Mode — discrete numeric values",
    "knn": "KNN — nearest-neighbour similarity",
    "iterative": "Iterative / MICE — model-based",
    "iterative_rf": "MissForest — random-forest MICE (slow)",
    "constant": "Constant + missing indicator",
    "interpolate": "Linear interpolation — ordered data",
    "ffill": "Forward fill — time series",
    "none": "None — only XGBoost/LightGBM/CatBoost",
}

CATEGORICAL_IMPUTERS = ["most_frequent", "constant"]
CATEGORICAL_IMPUTER_LABELS = {
    "most_frequent": "Mode — most frequent category",
    "constant": "Constant category (e.g. \"Missing\")",
}

# Models whose native implementations accept NaN directly, so ``none`` is safe.
NAN_CAPABLE_MODELS = {"xgboost", "lightgbm", "catboost"}

# ------------------------------------------------------------------------------
# Noise / outlier treatment strategies
# ------------------------------------------------------------------------------
NOISE_STRATEGIES = [
    "none",
    "rolling_mean",
    "rolling_median",
    "ema",
    "savgol",
    "hampel",
    "interpolate",
    "ffill",
    "iqr_clip",
    "zscore_clip",
]

NOISE_STRATEGY_LABELS = {
    "none": "None — leave features untouched",
    "rolling_mean": "Rolling mean — smooth moving average",
    "rolling_median": "Rolling median — resistant to spikes",
    "ema": "Exponential moving average",
    "savgol": "Savitzky–Golay — preserve peaks and shape",
    "hampel": "Hampel filter — replace spike outliers only",
    "interpolate": "Linear interpolation — fill gaps",
    "ffill": "Forward fill — carry last value",
    "iqr_clip": "IQR clip / winsorize — cap extreme tails",
    "zscore_clip": "Z-score clip — cap at μ ± kσ",
}

OUTLIER_METHODS = ["none", "iqr", "zscore", "isolation_forest"]
OUTLIER_METHOD_LABELS = {
    "none": "None — keep every row",
    "iqr": "IQR fence (Q1 − k·IQR, Q3 + k·IQR)",
    "zscore": "Z-score (|z| > k)",
    "isolation_forest": "Isolation Forest (multivariate)",
}

OUTLIER_ACTIONS = ["clip", "remove"]
OUTLIER_ACTION_LABELS = {
    "clip": "Clip values to the fence",
    "remove": "Remove flagged rows (training split only)",
}

INJECTION_METHODS = ["none", "gaussian", "spikes", "missing"]
INJECTION_METHOD_LABELS = {
    "none": "None — use the data as loaded",
    "gaussian": "Gaussian noise (σ = ratio × column std)",
    "spikes": "Random spikes (±kσ)",
    "missing": "Random missing cells (NaN)",
}

# ------------------------------------------------------------------------------
# AI interpretation (Groq Cloud, OpenAI-compatible endpoint)
# ------------------------------------------------------------------------------
AI_PROVIDER = "groq"
AI_BASE_URL = "https://api.groq.com/openai/v1"
AI_MODEL = "qwen/qwen3.8-27b"
AI_PROMPT_VERSION = "v1"
AI_TIMEOUT_SECONDS = 120
AI_MAX_CONTEXT_CHARS = 12000

