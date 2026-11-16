"""
Data preprocessing pipeline using scikit-learn ColumnTransformer.
Ensures zero data leakage by fitting strictly on training split.

Numeric imputation is pluggable (median, mean, mode, KNN, MICE, MissForest-style
random-forest MICE, constant + missing indicator, interpolation, forward fill or
none for NaN-native boosters). Categorical imputation supports mode and a
constant category. Fitted transformers remain the single source of truth for
test-time and live-inference transforms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from config import TASK_CLASSIFICATION


class _PandasInterpolator(BaseEstimator, TransformerMixin):
    """Linear interpolation / forward fill for ordered numeric columns.

    Used as a scikit-learn transformer so it lives inside the fitted
    ``ColumnTransformer``. Single-row live inference cannot interpolate, so the
    training medians captured at ``fit`` fill any remaining gap.
    """

    def __init__(self, method: str = "interpolate", columns: Optional[List[str]] = None):
        self.method = method
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> "_PandasInterpolator":
        frame = self._as_frame(X)
        self.columns_ = list(frame.columns)
        self.fallback_ = {
            column: (
                float(pd.to_numeric(frame[column], errors="coerce").median())
                if pd.to_numeric(frame[column], errors="coerce").notna().any()
                else 0.0
            )
            for column in self.columns_
        }
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        frame = X.copy() if isinstance(X, pd.DataFrame) else self._as_frame(X).copy()
        if not isinstance(X, pd.DataFrame) and hasattr(self, "columns_"):
            # ColumnTransformer handed us an array: restore the trained names.
            if frame.shape[1] == len(self.columns_):
                frame.columns = self.columns_
        for column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            if self.method == "ffill":
                series = series.ffill()
            else:
                series = series.interpolate(method="linear", limit_direction="both")
            fallback = self.fallback_.get(column, 0.0)
            frame[column] = series.fillna(fallback)
        return frame

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = input_features if input_features is not None else getattr(self, "columns_", [])
        return np.asarray([str(name) for name in names], dtype=object)

    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return pd.DataFrame(array)


class DataPreprocessor:
    """Preprocesses feature matrices and target vectors cleanly."""

    def __init__(
        self,
        scale_numeric: bool = True,
        numeric_imputer: str = "median",
        categorical_imputer: str = "most_frequent",
        imputer_params: Optional[Dict[str, Any]] = None,
    ):
        self.scale_numeric = scale_numeric
        self.numeric_imputer = (numeric_imputer or "median").lower()
        self.categorical_imputer = (categorical_imputer or "most_frequent").lower()
        self.imputer_params: Dict[str, Any] = dict(imputer_params or {})

        self.column_transformer: Optional[ColumnTransformer] = None
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.feature_names_out: List[str] = []
        self.label_encoder: Optional[LabelEncoder] = None
        self.classes_: Optional[np.ndarray] = None
        self.removed_rows: int = 0

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------
    def _numeric_steps(self) -> List[tuple]:
        strategy = self.numeric_imputer
        params = self.imputer_params
        steps: List[tuple] = []

        add_indicator = bool(params.get("add_indicator", strategy == "constant"))
        random_state = int(params.get("random_state", 42))

        if strategy == "none":
            pass
        elif strategy in ("mean", "median", "most_frequent"):
            steps.append(
                ("imputer", SimpleImputer(strategy=strategy, add_indicator=add_indicator))
            )
        elif strategy == "constant":
            fill_value = params.get("fill_value", 0)
            steps.append(
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=fill_value,
                        add_indicator=add_indicator,
                    ),
                )
            )
        elif strategy == "knn":
            # KNN distances need comparable scales; StandardScaler ignores NaN
            # in fit and propagates it in transform, so it is safe before the
            # imputer.
            steps.append(("knn_scaler", StandardScaler()))
            steps.append(
                (
                    "imputer",
                    KNNImputer(
                        n_neighbors=int(params.get("n_neighbors", 5)),
                        weights=str(params.get("weights", "uniform")),
                        add_indicator=add_indicator,
                    ),
                )
            )
        elif strategy in ("iterative", "iterative_rf"):
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            except ImportError as exc:  # pragma: no cover - scikit-learn dependent
                raise ImportError(
                    "IterativeImputer is unavailable in this scikit-learn build."
                ) from exc
            from sklearn.impute import IterativeImputer

            estimator = None
            if strategy == "iterative_rf":
                from sklearn.ensemble import RandomForestRegressor

                estimator = RandomForestRegressor(
                    n_estimators=int(params.get("rf_estimators", 50)),
                    max_depth=params.get("rf_max_depth"),
                    n_jobs=-1,
                    random_state=random_state,
                )
            steps.append(
                (
                    "imputer",
                    IterativeImputer(
                        estimator=estimator,
                        max_iter=int(params.get("max_iter", 10)),
                        random_state=random_state,
                        add_indicator=add_indicator,
                    ),
                )
            )
        elif strategy in ("interpolate", "ffill"):
            steps.append(("imputer", _PandasInterpolator(method=strategy)))
        else:
            raise ValueError(f"Unknown numeric imputation strategy: {strategy}")

        if self.scale_numeric:
            steps.append(("scaler", StandardScaler()))
        return steps

    def _categorical_steps(self) -> List[tuple]:
        strategy = self.categorical_imputer
        params = self.imputer_params
        if strategy == "constant":
            imputer = SimpleImputer(
                strategy="constant", fill_value=str(params.get("cat_fill_value", "Missing"))
            )
        else:
            imputer = SimpleImputer(strategy="most_frequent")
        return [
            ("imputer", imputer),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------
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
            num_steps = self._numeric_steps()
            if num_steps:
                transformers.append(("num", Pipeline(num_steps), self.num_cols))
            else:
                # Raw pass-through: keep the numeric columns as-is.
                transformers.append(("num", "passthrough", self.num_cols))

        if self.cat_cols:
            transformers.append(
                ("cat", Pipeline(self._categorical_steps()), self.cat_cols)
            )

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
            y_train_proc = self._coerce_regression_target(y_train, split="train")

        return np.asarray(X_train_proc, dtype=np.float64), y_train_proc

    @staticmethod
    def _coerce_regression_target(y: pd.Series, split: str = "train") -> np.ndarray:
        """Convert a regression target to float, with a clear error for text values."""
        numeric = pd.to_numeric(y, errors="coerce")
        invalid = numeric.isna() & pd.notna(y)
        if invalid.any():
            samples = y[invalid].astype(str).unique()[:3].tolist()
            raise ValueError(
                f"Regression target contains {int(invalid.sum())} non-numeric value(s) in the "
                f"{split} split (e.g. {samples}). Select a numeric target column or switch "
                "the task type to classification."
            )
        return np.asarray(numeric, dtype=np.float64)

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
                y_test_proc = self._coerce_regression_target(y_test, split="test")

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

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------
    def _get_feature_names(self) -> List[str]:
        names: List[str] = []
        if self.column_transformer is None:
            return names

        for name, trans, cols in self.column_transformer.transformers_:
            if name == "num":
                if isinstance(trans, Pipeline):
                    names.extend(self._numeric_feature_names(trans, cols))
                else:
                    names.extend(str(c) for c in cols)
            elif name == "cat":
                onehot = trans.named_steps.get("onehot") if isinstance(trans, Pipeline) else None
                if onehot is not None and hasattr(onehot, "get_feature_names_out"):
                    names.extend(onehot.get_feature_names_out(cols).tolist())
                else:
                    names.extend(str(c) for c in cols)
        return names

    @staticmethod
    def _numeric_feature_names(pipeline: Pipeline, cols: List[str]) -> List[str]:
        """Names after imputation, including any missing-indicator columns."""
        imputer = pipeline.named_steps.get("imputer")
        fallback = [str(c) for c in cols]
        if imputer is not None and hasattr(imputer, "get_feature_names_out"):
            try:
                generated = imputer.get_feature_names_out(cols)
                if len(generated) >= len(cols):
                    return [str(name) for name in generated]
            except Exception:
                pass
        return fallback
