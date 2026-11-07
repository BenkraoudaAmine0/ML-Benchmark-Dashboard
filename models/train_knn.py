"""
K-Nearest Neighbors (KNN) Trainer.
Supports KNeighborsClassifier (Classification) and KNeighborsRegressor (Regression).
"""

from __future__ import annotations

from typing import Any, Dict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from Modeling.config import CAT_CLASSICAL, TASK_CLASSIFICATION
from Modeling.core.base import BaseModelTrainer


class KNNTrainer(BaseModelTrainer):
    """Trainer for K-Nearest Neighbors."""

    model_id = "knn"
    display_name = "K-Nearest Neighbors"
    category = CAT_CLASSICAL
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        return {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "minkowski",
            "p": 2,
        }

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            return KNeighborsClassifier(**merged)
        else:
            return KNeighborsRegressor(**merged)

