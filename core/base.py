"""
Base Model Trainer Interface.
Every model family implements this base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class BaseModelTrainer(ABC):
    """Abstract base class for all model trainers in the pipeline."""

    model_id: str = "base"
    display_name: str = "Base Model"
    category: str = "classical"  # 'classical' or 'ensemble'
    supported_tasks: Tuple[str, ...] = ("classification", "regression")

    def __init__(self, task: str = "classification", params: Optional[Dict[str, Any]] = None):
        self.task = task
        self.params = params or {}
        self.model: Any = None
        self.is_fitted: bool = False
        self.train_time: float = 0.0

    @abstractmethod
    def build_model(self, task: str, params: Dict[str, Any]) -> Any:
        """Construct the underlying scikit-learn / ensemble estimator."""
        pass

    @abstractmethod
    def get_default_params(self, task: str) -> Dict[str, Any]:
        """Return sensible default hyperparameters for the given task."""
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "BaseModelTrainer":
        """Fit the underlying estimator."""
        if self.model is None:
            self.model = self.build_model(self.task, self.params)
        self.model.fit(X_train, y_train, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions for the test split."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"Model {self.display_name} has not been fitted yet.")
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> Optional[np.ndarray]:
        """Generate class probabilities if task is classification and model supports it."""
        if self.task != "classification":
            return None
        if hasattr(self.model, "predict_proba"):
            try:
                return self.model.predict_proba(X_test)
            except Exception:
                return None
        elif hasattr(self.model, "decision_function"):
            # Provide decision function as fallback if needed
            return None
        return None

    def get_feature_importances(self, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importances or linear coefficients if available."""
        if not self.is_fitted or self.model is None:
            return None

        importances: Optional[np.ndarray] = None
        if hasattr(self.model, "feature_importances_"):
            importances = np.asarray(self.model.feature_importances_)
        elif hasattr(self.model, "coef_"):
            coef = np.asarray(self.model.coef_)
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)

        if importances is not None and len(importances) == len(feature_names):
            # Normalize to sum to 1.0 if sum > 0
            s = float(np.sum(importances))
            if s > 0:
                normalized = (importances / s).tolist()
            else:
                normalized = importances.tolist()
            return dict(zip(feature_names, [round(float(v), 5) for v in normalized]))

        return None

