"""
Linear / Logistic Regression Trainer.
Provides baseline models: LinearRegression/Ridge (Regression) and LogisticRegression (Classification).
"""

from __future__ import annotations

from typing import Any, Dict
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from Modeling.config import CAT_CLASSICAL, TASK_CLASSIFICATION
from Modeling.core.base import BaseModelTrainer


class LinearModelTrainer(BaseModelTrainer):
    """Baseline linear models."""

    model_id = "linear"
    display_name = "Linear / Logistic Baseline"
    category = CAT_CLASSICAL
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        if task == TASK_CLASSIFICATION:
            return {"C": 1.0, "max_iter": 2000, "random_state": 42}
        else:
            return {"fit_intercept": True}

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            return LogisticRegression(**merged)
        else:
            return LinearRegression(**merged)

