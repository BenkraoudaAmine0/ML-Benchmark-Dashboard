"""
Central Model Registry.
Coordinates model trainers, categories, and dependency verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from Modeling.core.base import BaseModelTrainer
from Modeling.config import CAT_CLASSICAL, CAT_ENSEMBLE, TASK_CLASSIFICATION, TASK_REGRESSION

# Import model trainers
from Modeling.models.train_linear import LinearModelTrainer
from Modeling.models.train_svm import SVMTrainer
from Modeling.models.train_knn import KNNTrainer
from Modeling.models.train_dtree import DecisionTreeTrainer
from Modeling.models.train_rf import RandomForestTrainer
from Modeling.models.train_adaboost import AdaBoostTrainer
from Modeling.models.train_xgboost import XGBoostTrainer
from Modeling.models.train_lightgbm import LightGBMTrainer
from Modeling.models.train_catboost import CatBoostTrainer
from Modeling.models.train_bagging import BaggingTrainer
from Modeling.models.train_stacking import StackingTrainer
from Modeling.models.train_voting import VotingTrainer


# Ordered dictionary of model registry
MODEL_REGISTRY: Dict[str, Type[BaseModelTrainer]] = {
    # Baseline
    "linear": LinearModelTrainer,
    # Classical Models
    "svm": SVMTrainer,
    "knn": KNNTrainer,
    "dtree": DecisionTreeTrainer,
    # Core Ensemble Models
    "rf": RandomForestTrainer,
    "adaboost": AdaBoostTrainer,
    "xgboost": XGBoostTrainer,
    "lightgbm": LightGBMTrainer,
    "catboost": CatBoostTrainer,
    # Advanced Ensemble Techniques
    "bagging": BaggingTrainer,
    "stacking": StackingTrainer,
    "voting": VotingTrainer,
}

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
