"""
Voting / Blending Ensemble Trainer.
Supports VotingClassifier (Soft/Hard Voting) and VotingRegressor.
"""

from __future__ import annotations

from typing import Any, Dict
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from Modeling.config import CAT_ENSEMBLE, TASK_CLASSIFICATION
from Modeling.core.base import BaseModelTrainer


class VotingTrainer(BaseModelTrainer):
    """Trainer for Voting / Blending ensembles."""

    model_id = "voting"
    display_name = "Voting / Blending Ensemble"
    category = CAT_ENSEMBLE
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        params = {"n_jobs": -1}
        if task == TASK_CLASSIFICATION:
            params["voting"] = "soft"
        return params

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            estimators = [
                ("knn", KNeighborsClassifier(n_neighbors=5)),
                ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)),
            ]
            return VotingClassifier(estimators=estimators, **merged)
        else:
            estimators = [
                ("knn", KNeighborsRegressor(n_neighbors=5)),
                ("dt", DecisionTreeRegressor(max_depth=5, random_state=42)),
                ("rf", RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)),
            ]
            return VotingRegressor(estimators=estimators, **merged)

