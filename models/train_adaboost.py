"""
AdaBoost Trainer (Ensemble / Boosting).
Supports AdaBoostClassifier (Classification) and AdaBoostRegressor (Regression).
"""

from __future__ import annotations

from typing import Any, Dict
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from config import CAT_ENSEMBLE, TASK_CLASSIFICATION
from core.base import BaseModelTrainer


class AdaBoostTrainer(BaseModelTrainer):
    """Trainer for AdaBoost ensemble."""

    model_id = "adaboost"
    display_name = "AdaBoost"
    category = CAT_ENSEMBLE
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        return {
            "n_estimators": 50,
            "learning_rate": 1.0,
            "random_state": 42,
        }

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            return AdaBoostClassifier(**merged)
        else:
            return AdaBoostRegressor(**merged)

