"""
CatBoost Trainer (Categorical Boosting).
Supports CatBoostClassifier (Classification) and CatBoostRegressor (Regression).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

from config import CAT_ENSEMBLE, TASK_CLASSIFICATION
from core.base import BaseModelTrainer


class CatBoostTrainer(BaseModelTrainer):
    """Trainer for CatBoost."""

    model_id = "catboost"
    display_name = "CatBoost"
    category = CAT_ENSEMBLE
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        return {
            "iterations": 150,
            "learning_rate": 0.1,
            "depth": 6,
            "random_seed": 42,
            "verbose": 0,
        }

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            return CatBoostClassifier(**merged)
        else:
            return CatBoostRegressor(**merged)

    def get_feature_importances(self, feature_names: List[str]) -> Optional[Dict[str, float]]:
        if not self.is_fitted or self.model is None:
            return None
        try:
            if hasattr(self.model, "get_feature_importance"):
                importances = np.asarray(self.model.get_feature_importance())
                s = float(np.sum(importances))
                if s > 0:
                    normalized = (importances / s).tolist()
                else:
                    normalized = importances.tolist()
                return dict(zip(feature_names, [round(float(v), 5) for v in normalized]))
        except Exception:
            pass
        return super().get_feature_importances(feature_names)

