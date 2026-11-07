"""
XGBoost Trainer (Extreme Gradient Boosting).
Supports XGBClassifier (Classification) and XGBRegressor (Regression).
"""

from __future__ import annotations

from typing import Any, Dict
import xgboost as xgb

from Modeling.config import CAT_ENSEMBLE, TASK_CLASSIFICATION
from Modeling.core.base import BaseModelTrainer


class XGBoostTrainer(BaseModelTrainer):
    """Trainer for XGBoost."""

    model_id = "xgboost"
    display_name = "XGBoost"
    category = CAT_ENSEMBLE
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
        if task == TASK_CLASSIFICATION:
            params["eval_metric"] = "logloss"
        else:
            params["eval_metric"] = "rmse"
        return params

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            return xgb.XGBClassifier(**merged)
        else:
            return xgb.XGBRegressor(**merged)

