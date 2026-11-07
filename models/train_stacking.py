"""
Stacking Ensemble Trainer.
Supports StackingClassifier (Classification) and StackingRegressor (Regression).
"""

from __future__ import annotations

from typing import Any, Dict
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from Modeling.config import CAT_ENSEMBLE, TASK_CLASSIFICATION
from Modeling.core.base import BaseModelTrainer


class StackingTrainer(BaseModelTrainer):
    """Trainer for Stacking ensembles with cross-validated meta-learner."""

    model_id = "stacking"
    display_name = "Stacking (Meta-Learner)"
    category = CAT_ENSEMBLE
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        return {
            "cv": 5,
            "n_jobs": -1,
        }

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            estimators = [
                ("knn", KNeighborsClassifier(n_neighbors=5)),
                ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)),
            ]
            final_estimator = LogisticRegression(max_iter=1000)
            return StackingClassifier(
                estimators=estimators, final_estimator=final_estimator, **merged
            )
        else:
            estimators = [
                ("knn", KNeighborsRegressor(n_neighbors=5)),
                ("dt", DecisionTreeRegressor(max_depth=5, random_state=42)),
                ("rf", RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)),
            ]
            final_estimator = LinearRegression()
            return StackingRegressor(
                estimators=estimators, final_estimator=final_estimator, **merged
            )

