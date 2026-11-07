"""
Bagging Ensemble Trainer.
Supports BaggingClassifier (Classification) and BaggingRegressor (Regression).
"""

from __future__ import annotations

from typing import Any, Dict
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from Modeling.config import CAT_ENSEMBLE, TASK_CLASSIFICATION
from Modeling.core.base import BaseModelTrainer


class BaggingTrainer(BaseModelTrainer):
    """Trainer for Bagging (Bootstrap Aggregation) ensembles."""

    model_id = "bagging"
    display_name = "Bagging (Trees)"
    category = CAT_ENSEMBLE
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        return {
            "n_estimators": 50,
            "max_samples": 1.0,
            "max_features": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            estimator = DecisionTreeClassifier(random_state=42)
            return BaggingClassifier(estimator=estimator, **merged)
        else:
            estimator = DecisionTreeRegressor(random_state=42)
            return BaggingRegressor(estimator=estimator, **merged)

