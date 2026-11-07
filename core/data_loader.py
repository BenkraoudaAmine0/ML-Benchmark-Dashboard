"""
Dataset loader, discovery, task detection, and splitting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Modeling.config import DATASETS_DIR, TASK_CLASSIFICATION, TASK_REGRESSION


def discover_datasets(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Scan the Datasets folder recursively for valid CSV datasets.
    Returns a dict mapping display names to Path objects.
    """
    directory = base_dir or DATASETS_DIR
    datasets: Dict[str, Path] = {}
    if not directory.exists():
        return datasets

    for path in directory.rglob("*.csv"):
        # Ignore temporary lock files
        if path.name.startswith("~") or path.name.startswith("."):
            continue
        try:
            rel = path.relative_to(directory)
            label = str(rel).replace("\\", "/")
        except ValueError:
            label = path.name
        datasets[label] = path

    return dict(sorted(datasets.items()))


def load_dataset(file_source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    if isinstance(file_source, pd.DataFrame):
        return file_source.copy()

    path = Path(file_source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    for enc in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue

    return pd.read_csv(path)


def detect_task_type(y: pd.Series) -> str:
    """
    Infer whether the target variable corresponds to classification or regression.
    """
    # Non-numeric is always classification
    if not pd.api.types.is_numeric_dtype(y):
        return TASK_CLASSIFICATION

    # Drop missing values for cardinality check
    y_clean = y.dropna()
    n_unique = y_clean.nunique()

    # Float values with high cardinality -> Regression
    if pd.api.types.is_float_dtype(y) and n_unique > 20:
        return TASK_REGRESSION

    # Integer values with low cardinality (<= 15) -> Classification
    if n_unique <= 15:
        return TASK_CLASSIFICATION

    return TASK_REGRESSION


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    task: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_cols: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str, List[Any]]:
    """
    Prepares dataset by dropping unnecessary columns, removing target NaNs,
    subsampling if max_samples is provided, and performing train-test split.

    Returns:
        (X_train, X_test, y_train, y_test, inferred_or_confirmed_task, classes_list)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop specified columns and rows where target is null
    df_clean = df.copy()
    if drop_cols:
        existing_drops = [c for c in drop_cols if c in df_clean.columns and c != target_col]
        df_clean = df_clean.drop(columns=existing_drops)

    df_clean = df_clean.dropna(subset=[target_col]).reset_index(drop=True)

    # Subsample if dataset is larger than requested max_samples
    if max_samples is not None and max_samples > 0 and len(df_clean) > max_samples:
        df_clean = df_clean.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])

    # Infer or confirm task
    task_type = task if task in (TASK_CLASSIFICATION, TASK_REGRESSION) else detect_task_type(y)

    classes_list: List[Any] = []
    stratify = None

    if task_type == TASK_CLASSIFICATION:
        # Check if every class has at least 2 samples for stratification
        class_counts = y.value_counts()
        classes_list = sorted(y.unique().tolist(), key=lambda x: str(x))
        if (class_counts >= 2).all() and len(class_counts) > 1:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    return X_train, X_test, y_train, y_test, task_type, classes_list

