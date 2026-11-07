"""
Decision Tree Trainer.
Supports DecisionTreeClassifier (Classification) and DecisionTreeRegressor (Regression).
"""

from __future__ import annotations

from typing import Any, Dict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from Modeling.config import CAT_CLASSICAL, TASK_CLASSIFICATION
from Modeling.core.base import BaseModelTrainer


class DecisionTreeTrainer(BaseModelTrainer):
    """Trainer for Decision Trees."""

    model_id = "dtree"
    display_name = "Decision Tree"
    category = CAT_CLASSICAL
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        params = {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
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
            return DecisionTreeClassifier(**merged)
        else:
            return DecisionTreeRegressor(**merged)

