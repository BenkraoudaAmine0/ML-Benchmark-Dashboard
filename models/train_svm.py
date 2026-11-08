"""
Support Vector Machine (SVM) Trainer.
Supports both SVC (Classification) and SVR (Regression).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict
from sklearn.svm import SVC, SVR

# Suppress scikit-learn 1.9+ probability deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from config import CAT_CLASSICAL, TASK_CLASSIFICATION
from core.base import BaseModelTrainer


class SVMTrainer(BaseModelTrainer):
    """Trainer for Support Vector Machines (SVC and SVR)."""

    model_id = "svm"
    display_name = "Support Vector Machine"
    category = CAT_CLASSICAL
    supported_tasks = ("classification", "regression")

    def get_default_params(self, task: str) -> Dict[str, Any]:
        params = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
        }
        if task == TASK_CLASSIFICATION:
            params["probability"] = True
        else:
            params["epsilon"] = 0.1
        return params

    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        merged = self.get_default_params(task)
        merged.update(params)

        if task == TASK_CLASSIFICATION:
            merged["probability"] = True
            merged["random_state"] = 42
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                return SVC(**merged)
        else:
            return SVR(**merged)
