"""
Random Forest Trainer (Ensemble / Bagging).
Supports RandomForestClassifier (Classification) and RandomForestRegressor (Regression).
"""

from __future__ import annotations

from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from Modeling.config import CAT_ENSEMBLE, TASK_CLASSIFICATION
from Modeling.core.base import BaseModelTrainer


class RandomForestTrainer(BaseModelTrainer):
    """Trainer for Random Forest ensembles."""

    model_id = "rf"
    display_name = "Random Forest"
    category = CAT_ENSEMBLE
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": 42,
        }
        if task == TASK_CLASSIFICATION:
            params["criterion"] = "gini"
        else:
            params["criterion"] = "squared_error"
        return params

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            return RandomForestClassifier(**merged)
        else:
            return RandomForestRegressor(**merged)

