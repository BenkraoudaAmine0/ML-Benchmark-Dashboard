"""
Training Pipeline Orchestrator.
Coordinates data loading, preprocessing, model training, evaluation, and SQLite persistence.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Suppress sklearn future warnings during fitting
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from config import (
    CAT_CLASSICAL,
    CAT_ENSEMBLE,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
)
from core.data_loader import prepare_data
from core.metrics import evaluate_model
from core.preprocessor import DataPreprocessor
from core.storage import save_experiment, save_model_result
from models.registry import MODEL_REGISTRY, instantiate_trainer


class TrainingPipeline:
    """Manages full benchmark and training pipeline runs."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        task: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        drop_cols: Optional[List[str]] = None,
        scale_numeric: bool = True,
        max_samples: Optional[int] = None,
        **kwargs,
    ):
        self.df = df
        self.target_col = target_col
        self.task = task
        self.test_size = test_size
        self.random_state = random_state
        self.drop_cols = drop_cols or []
        self.scale_numeric = scale_numeric
        self.max_samples = max_samples or kwargs.get("max_samples")

        # Prepared data components
        self.X_train_raw: Optional[pd.DataFrame] = None
        self.X_test_raw: Optional[pd.DataFrame] = None
        self.y_train_raw: Optional[pd.Series] = None
        self.y_test_raw: Optional[pd.Series] = None
        self.classes_list: List[Any] = []
        self.task_type: str = ""

        # Preprocessor and processed matrices
        self.preprocessor = DataPreprocessor(scale_numeric=scale_numeric)
        self.X_train_proc: Optional[np.ndarray] = None
        self.X_test_proc: Optional[np.ndarray] = None
        self.y_train_proc: Optional[np.ndarray] = None
        self.y_test_proc: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

        # Fitted trainers cache (in-memory for active session)
        self.fitted_trainers: Dict[str, Any] = {}

    def prepare(self) -> None:
        """Split and preprocess the dataset."""
        (
            self.X_train_raw,
            self.X_test_raw,
            self.y_train_raw,
            self.y_test_raw,
            self.task_type,
            self.classes_list,
        ) = prepare_data(
            df=self.df,
            target_col=self.target_col,
            task=self.task,
            test_size=self.test_size,
            random_state=self.random_state,
            drop_cols=self.drop_cols,
            max_samples=self.max_samples,
        )

        # Fit preprocessor
        self.X_train_proc, self.y_train_proc = self.preprocessor.fit_transform(
            self.X_train_raw, self.y_train_raw, task=self.task_type
        )
        self.X_test_proc, self.y_test_proc = self.preprocessor.transform(
            self.X_test_raw, self.y_test_raw, task=self.task_type
        )
        self.feature_names = self.preprocessor.feature_names_out

    def run_benchmark(
        self,
        model_ids: Optional[List[str]] = None,
        custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
        run_name: Optional[str] = None,
        dataset_name: str = "Dataset",
        progress_cb: Optional[Callable[[str, float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute training and evaluation for the selected models.
        Saves run and benchmark records directly to SQLite.
        """
        if self.X_train_proc is None:
            self.prepare()

        selected_ids = model_ids or list(MODEL_REGISTRY.keys())
        custom_params = custom_params or {}

        # 1. Initialize Experiment in DB
        exp_name = run_name or f"{dataset_name}_{self.task_type}_{int(time.time())}"
        experiment_id = save_experiment(
            run_name=exp_name,
            dataset_name=dataset_name,
            task_type=self.task_type,
            target_col=self.target_col,
            n_train=len(self.X_train_proc),
            n_test=len(self.X_test_proc),
            n_features=len(self.feature_names),
            feature_names=self.feature_names,
        )

        results_list = []
        total_models = len(selected_ids)

        for idx, mid in enumerate(selected_ids):
            if mid not in MODEL_REGISTRY:
                continue

            pct = (idx) / max(1, total_models)
            if progress_cb:
                progress_cb(mid, pct, f"Training {MODEL_REGISTRY[mid].display_name}...")

            trainer = instantiate_trainer(
                model_id=mid,
                task=self.task_type,
                params=custom_params.get(mid),
            )

            # Fit with timer
            t0 = time.perf_counter()
            try:
                trainer.fit(self.X_train_proc, self.y_train_proc)
                t1 = time.perf_counter()
                elapsed = round(t1 - t0, 4)

                # Predictions
                y_pred = trainer.predict(self.X_test_proc)
                y_prob = (
                    trainer.predict_proba(self.X_test_proc)
                    if self.task_type == TASK_CLASSIFICATION
                    else None
                )

                # Metrics
                class_names = (
                    [str(c) for c in self.preprocessor.classes_]
                    if self.preprocessor.classes_ is not None
                    else None
                )
                metrics = evaluate_model(
                    task=self.task_type,
                    y_true=self.y_test_proc,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    class_names=class_names,
                )

                # Compute Train Metrics & Overfitting Diagnostics
                try:
                    y_train_pred = trainer.predict(self.X_train_proc)
                    if self.task_type == TASK_CLASSIFICATION:
                        from sklearn.metrics import accuracy_score, f1_score
                        tr_f1 = round(float(f1_score(self.y_train_proc, y_train_pred, average="macro", zero_division=0)), 4)
                        tr_acc = round(float(accuracy_score(self.y_train_proc, y_train_pred)), 4)
                        metrics["train_f1_macro"] = tr_f1
                        metrics["train_accuracy"] = tr_acc
                        gap = round(tr_f1 - float(metrics.get("f1_macro") or 0.0), 4)
                        metrics["overfit_gap"] = gap
                        metrics["is_overfitting"] = bool(gap >= 0.15)
                    else:
                        from sklearn.metrics import mean_squared_error, r2_score
                        tr_r2 = round(float(r2_score(self.y_train_proc, y_train_pred)), 4)
                        tr_rmse = round(float(np.sqrt(mean_squared_error(self.y_train_proc, y_train_pred))), 4)
                        metrics["train_r2"] = tr_r2
                        metrics["train_rmse"] = tr_rmse
                        gap = round(tr_r2 - float(metrics.get("r2") or 0.0), 4)
                        metrics["overfit_gap"] = gap
                        metrics["is_overfitting"] = bool(gap >= 0.15)
                except Exception:
                    metrics["is_overfitting"] = False

                # Feature importances
                importances = trainer.get_feature_importances(self.feature_names)

                # Save trained model artifact to disk
                import joblib
                from config import ARTIFACTS_DIR
                artifact_filename = f"exp_{experiment_id}_{mid}.joblib"
                artifact_path = ARTIFACTS_DIR / artifact_filename
                try:
                    joblib.dump({"model": trainer.model, "preprocessor": self.preprocessor}, artifact_path)
                    metrics["artifact_path"] = str(artifact_path)
                except Exception:
                    pass

                # Predictions sample for dashboard visualization (up to 50 items)
                n_sample = min(50, len(self.y_test_proc))
                actual_sample = self.y_test_proc[:n_sample].tolist()
                pred_sample = y_pred[:n_sample].tolist()

                # Revert classification labels for readability
                if self.task_type == TASK_CLASSIFICATION and self.preprocessor.label_encoder:
                    try:
                        actual_sample = self.preprocessor.inverse_transform_target(
                            self.y_test_proc[:n_sample]
                        ).tolist()
                        pred_sample = self.preprocessor.inverse_transform_target(
                            y_pred[:n_sample]
                        ).tolist()
                    except Exception:
                        pass

                predictions_sample = {
                    "actual": actual_sample,
                    "predicted": pred_sample,
                }

                # Save to SQLite
                res_id = save_model_result(
                    experiment_id=experiment_id,
                    model_id=mid,
                    model_name=trainer.display_name,
                    category=trainer.category,
                    train_time_sec=elapsed,
                    metrics=metrics,
                    params=trainer.params,
                    feature_importances=importances,
                    predictions_sample=predictions_sample,
                )

                self.fitted_trainers[mid] = trainer

                results_list.append(
                    {
                        "result_id": res_id,
                        "model_id": mid,
                        "model_name": trainer.display_name,
                        "category": trainer.category,
                        "train_time_sec": elapsed,
                        "metrics": metrics,
                        "feature_importances": importances,
                    }
                )

            except Exception as e:
                # Log error and continue to other models
                print(f"Error training {mid}: {e}")

        if progress_cb:
            progress_cb("done", 1.0, "Training completed!")

        return {
            "experiment_id": experiment_id,
            "run_name": exp_name,
            "task_type": self.task_type,
            "results": results_list,
            "n_train": len(self.X_train_proc),
            "n_test": len(self.X_test_proc),
        }

