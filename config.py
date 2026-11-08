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

