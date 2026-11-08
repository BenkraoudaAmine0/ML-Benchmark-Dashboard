"""
Central Model Registry.
Coordinates model trainers, categories, and dependency verification.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Type

from core.base import BaseModelTrainer
from config import CAT_CLASSICAL, CAT_ENSEMBLE, TASK_CLASSIFICATION, TASK_REGRESSION

logger = logging.getLogger(__name__)

# Map of trainer class name -> pip requirement for trainers that could not be
# imported because an optional dependency is not installed.
MISSING_DEPENDENCIES: Dict[str, str] = {}


def _load_trainer(module: str, class_name: str, requirement: str) -> Optional[type]:
    """Import a trainer class, tolerating a missing optional dependency."""
    try:
        return getattr(importlib.import_module(module), class_name)
    except ImportError as exc:
        MISSING_DEPENDENCIES[class_name] = requirement
        logger.warning("Skipping %s (install %r): %s", class_name, requirement, exc)
        return None


# (model id, module, class name, required package)
_TRAINER_IMPORTS = [
    # Baseline
    ("linear", "models.train_linear", "LinearModelTrainer", "scikit-learn"),
    # Classical Models
    ("svm", "models.train_svm", "SVMTrainer", "scikit-learn"),
    ("knn", "models.train_knn", "KNNTrainer", "scikit-learn"),
    ("dtree", "models.train_dtree", "DecisionTreeTrainer", "scikit-learn"),
    # Core Ensemble Models
    ("rf", "models.train_rf", "RandomForestTrainer", "scikit-learn"),
    ("adaboost", "models.train_adaboost", "AdaBoostTrainer", "scikit-learn"),
    ("xgboost", "models.train_xgboost", "XGBoostTrainer", "xgboost"),
    ("lightgbm", "models.train_lightgbm", "LightGBMTrainer", "lightgbm"),
    ("catboost", "models.train_catboost", "CatBoostTrainer", "catboost"),
    # Advanced Ensemble Techniques
    ("bagging", "models.train_bagging", "BaggingTrainer", "scikit-learn"),
    ("stacking", "models.train_stacking", "StackingTrainer", "scikit-learn"),
    ("voting", "models.train_voting", "VotingTrainer", "scikit-learn"),
]

# Ordered dictionary of model registry (only trainers whose dependencies exist)
MODEL_REGISTRY: Dict[str, Type[BaseModelTrainer]] = {}
for _model_id, _module, _class_name, _requirement in _TRAINER_IMPORTS:
    _trainer_cls = _load_trainer(_module, _class_name, _requirement)
    if _trainer_cls is not None:
        MODEL_REGISTRY[_model_id] = _trainer_cls

# The 8 Core Models explicitly requested
CORE_8_MODEL_IDS = ["svm", "knn", "dtree", "rf", "adaboost", "xgboost", "lightgbm", "catboost"]


def get_trainer_class(model_id: str) -> Type[BaseModelTrainer]:
    """Retrieve trainer class by ID."""
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model id: {model_id}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id]


def instantiate_trainer(
    model_id: str,
    task: str = TASK_CLASSIFICATION,
    params: Optional[Dict[str, Any]] = None,
) -> BaseModelTrainer:
    """Create a new trainer instance."""
    cls = get_trainer_class(model_id)
    return cls(task=task, params=params)


def list_available_models(task: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List metadata for all registered models.
    Optionally filter by task support.
    """
    items = []
    for mid, cls in MODEL_REGISTRY.items():
        if task and task not in cls.supported_tasks:
            continue
        dummy = cls(task=task or TASK_CLASSIFICATION)
        items.append(
            {
                "id": mid,
                "name": cls.display_name,
                "category": cls.category,
                "is_core": mid in CORE_8_MODEL_IDS,
                "supported_tasks": list(cls.supported_tasks),
                "default_params": dummy.get_default_params(task or TASK_CLASSIFICATION),
            }
        )
    return items
