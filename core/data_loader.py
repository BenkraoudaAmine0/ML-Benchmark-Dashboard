"""
Dataset loader, discovery, task detection, and splitting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import DATASETS_DIR, TASK_CLASSIFICATION, TASK_REGRESSION
from core.splits import SAMPLING_RANDOM, split_or_create


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
    *,
    sampling_strategy: str = SAMPLING_RANDOM,
    split_id: Optional[int] = None,
    save_split: bool = True,
    allow_reuse: bool = True,
    dataset_name: str = "Dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str, List[Any]]:
    """
    Prepares the dataset through the persistent split layer.

    Dropping columns/rows, subsampling and the train-test split are delegated to
    :func:`core.splits.split_or_create`, which reuses an identical saved split
    when one exists.

    Returns:
        (X_train, X_test, y_train, y_test, inferred_or_confirmed_task, classes_list)
    """
    split = split_or_create(
        df=df,
        target_col=target_col,
        task=task,
        test_size=test_size,
        random_state=random_state,
        drop_cols=drop_cols,
        max_samples=max_samples,
        sampling_strategy=sampling_strategy,
        dataset_name=dataset_name,
        split_id=split_id,
        save=save_split,
        allow_reuse=allow_reuse,
    )
    return (
        split.X_train,
        split.X_test,
        split.y_train,
        split.y_test,
        split.task_type,
        split.classes,
    )

