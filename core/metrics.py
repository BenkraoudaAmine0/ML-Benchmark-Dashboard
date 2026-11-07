"""
Evaluation metrics suite for Classification and Regression tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from Modeling.config import TASK_CLASSIFICATION, TASK_REGRESSION


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics."""
    metrics: Dict[str, Any] = {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Basic scores
    metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
    metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(y_true, y_pred)), 4)
    metrics["f1_macro"] = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    metrics["f1_weighted"] = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
    metrics["precision_macro"] = round(
        float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
    )
    metrics["recall_macro"] = round(
        float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
    )

    # Probabilistic metrics
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary ROC-AUC
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, prob_pos)), 4)
            elif n_classes > 2 and y_prob.ndim == 2 and y_prob.shape[1] == n_classes:
                # Multiclass OvR ROC-AUC
                metrics["roc_auc"] = round(
                    float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")), 4
                )
            metrics["log_loss"] = round(float(log_loss(y_true, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None
            metrics["log_loss"] = None
    else:
        metrics["roc_auc"] = None
        metrics["log_loss"] = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Detailed report
    try:
        target_names = [str(c) for c in class_names] if class_names else None
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )
    except Exception:
        metrics["classification_report"] = {}

    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute comprehensive regression performance metrics."""
    metrics: Dict[str, Any] = {}

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mx_err = float(max_error(y_true, y_pred))

    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        mape = None

    residuals = (y_true - y_pred).tolist()

    metrics["r2"] = round(r2, 4)
    metrics["rmse"] = round(rmse, 4)
    metrics["mae"] = round(mae, 4)
    metrics["mse"] = round(mse, 4)
    metrics["median_ae"] = round(medae, 4)
    metrics["max_error"] = round(mx_err, 4)
    metrics["mape"] = round(mape, 4) if mape is not None else None
    metrics["explained_variance"] = round(evs, 4)
    metrics["residuals_sample"] = residuals[:100]  # Store sample for plotting
    metrics["residual_mean"] = round(float(np.mean(residuals)), 4)
    metrics["residual_std"] = round(float(np.std(residuals)), 4)

    return metrics


def evaluate_model(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Universal dispatcher for model evaluation."""
    if task == TASK_CLASSIFICATION:
        return evaluate_classification(y_true, y_pred, y_prob, class_names)
    elif task == TASK_REGRESSION:
        return evaluate_regression(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported task: {task}")

