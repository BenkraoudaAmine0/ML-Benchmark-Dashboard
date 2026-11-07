"""
Configuration and constants for ML2 Training Pipeline.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
STORAGE_DIR = BASE_DIR / "storage"
DB_PATH = STORAGE_DIR / "training_results.db"
ARTIFACTS_DIR = STORAGE_DIR / "artifacts"

# Ensure storage directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Supported tasks
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
SUPPORTED_TASKS = [TASK_CLASSIFICATION, TASK_REGRESSION]

# Model categories
CAT_CLASSICAL = "classical"
CAT_ENSEMBLE = "ensemble"

