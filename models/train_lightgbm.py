"""
LightGBM Trainer (Light Gradient Boosting Machine).
Supports LGBMClassifier (Classification) and LGBMRegressor (Regression).
"""

from __future__ import annotations

from typing import Any, Dict
import lightgbm as lgb

from Modeling.config import CAT_ENSEMBLE, TASK_CLASSIFICATION
from Modeling.core.base import BaseModelTrainer


class LightGBMTrainer(BaseModelTrainer):
    """Trainer for LightGBM."""

    model_id = "lightgbm"
    display_name = "LightGBM"
    category = CAT_ENSEMBLE
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            return lgb.LGBMClassifier(**merged)
        else:
            return lgb.LGBMRegressor(**merged)

