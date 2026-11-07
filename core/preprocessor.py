"""
Data preprocessing pipeline using scikit-learn ColumnTransformer.
Ensures zero data leakage by fitting strictly on training split.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from Modeling.config import TASK_CLASSIFICATION


class DataPreprocessor:
    """Preprocesses feature matrices and target vectors cleanly."""

    def __init__(self, scale_numeric: bool = True):
        self.scale_numeric = scale_numeric
        self.column_transformer: Optional[ColumnTransformer] = None
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.feature_names_out: List[str] = []
        self.label_encoder: Optional[LabelEncoder] = None
        self.classes_: Optional[np.ndarray] = None

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor on train data and return transformed arrays."""
        # Detect numeric vs categorical
        self.num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = [c for c in X_train.columns if c not in self.num_cols]

        transformers = []

        if self.num_cols:
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if self.scale_numeric:
                num_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(num_steps), self.num_cols))

        if self.cat_cols:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
            transformers.append(("cat", Pipeline(cat_steps), self.cat_cols))

        if not transformers:
            raise ValueError("Input dataframe has no numeric or categorical columns.")

        self.column_transformer = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )
        X_train_proc = self.column_transformer.fit_transform(X_train)

        # Extract generated feature names
        self.feature_names_out = self._get_feature_names()

        # Handle Target y
        if task == TASK_CLASSIFICATION:
            self.label_encoder = LabelEncoder()
            y_train_proc = self.label_encoder.fit_transform(y_train)
            self.classes_ = self.label_encoder.classes_
        else:
            self.label_encoder = None
            self.classes_ = None
            y_train_proc = np.asarray(y_train, dtype=np.float64)

        return np.asarray(X_train_proc, dtype=np.float64), y_train_proc

    def transform(
        self,
        X_test: pd.DataFrame,
        y_test: Optional[pd.Series] = None,
        task: str = TASK_CLASSIFICATION,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform test or unseen data using the fitted preprocessor."""
        if self.column_transformer is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")

        X_test_proc = self.column_transformer.transform(X_test)

        y_test_proc = None
        if y_test is not None:
            if task == TASK_CLASSIFICATION:
                if self.label_encoder is not None:
                    # In case test has unseen labels, handle gracefully
                    known_classes = set(self.label_encoder.classes_)
                    y_test_safe = y_test.map(
                        lambda val: val if val in known_classes else self.label_encoder.classes_[0]
                    )
                    y_test_proc = self.label_encoder.transform(y_test_safe)
                else:
                    y_test_proc = np.asarray(y_test)
            else:
                y_test_proc = np.asarray(y_test, dtype=np.float64)

        return np.asarray(X_test_proc, dtype=np.float64), y_test_proc

    def transform_single_dict(self, sample_dict: dict) -> np.ndarray:
        """Transform a single input sample dictionary for live inference."""
        if self.column_transformer is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        df_sample = pd.DataFrame([sample_dict])
        return np.asarray(self.column_transformer.transform(df_sample), dtype=np.float64)

    def inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        """Convert encoded classification labels back to original format."""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(np.asarray(y_pred, dtype=int))
        return y_pred

    def _get_feature_names(self) -> List[str]:
        names: List[str] = []
        if self.column_transformer is None:
            return names

        for name, trans, cols in self.column_transformer.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                onehot = trans.named_steps.get("onehot")
                if onehot is not None and hasattr(onehot, "get_feature_names_out"):
                    names.extend(onehot.get_feature_names_out(cols).tolist())
                else:
                    names.extend(cols)
        return names

