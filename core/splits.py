"""
Reproducible dataset splitting with persistent, content-addressed partitions.

A split is identified by a fingerprint of the *cleaned* dataset content plus the
split settings (target, test size, seed, subsampling, dropped columns). Re-running
training with the same fingerprint reloads the exact partitions from disk, so
results never drift because a split was re-drawn.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import SPLITS_DIR, TASK_CLASSIFICATION, TASK_REGRESSION
from core.storage import (
    find_split_by_fingerprint,
    get_split,
    save_split_record,
)

SAMPLING_RANDOM = "random"
SAMPLING_FIRST = "first"


@dataclass
class SplitResult:
    """Partitions plus the metadata needed to display and re-use them."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    task_type: str
    classes: List[Any] = field(default_factory=list)
    split_id: Optional[int] = None
    fingerprint: str = ""
    created: bool = False
    reused: bool = False
    n_rows: int = 0
    n_train: int = 0
    n_test: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    source: str = "created"
    # Cleaned, subsampled source frame (in original order) for data profiling.
    # Never persisted; reconstructed from the partitions when a split is loaded.
    clean_frame: Optional[pd.DataFrame] = None


# ------------------------------------------------------------------------------
# Cleaning / sampling
# ------------------------------------------------------------------------------
def clean_dataframe(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Drop requested columns and rows without a target value."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    cleaned = df.copy()
    if drop_cols:
        existing = [c for c in drop_cols if c in cleaned.columns and c != target_col]
        cleaned = cleaned.drop(columns=existing)
    cleaned = cleaned.dropna(subset=[target_col]).reset_index(drop=True)
    return cleaned


def subsample(
    df: pd.DataFrame,
    max_samples: Optional[int],
    random_state: int = 42,
    sampling_strategy: str = SAMPLING_RANDOM,
) -> pd.DataFrame:
    """Limit the row count either randomly (seeded) or by original order."""
    if not max_samples or max_samples <= 0 or len(df) <= max_samples:
        return df
    if sampling_strategy == SAMPLING_FIRST:
        return df.iloc[:max_samples].reset_index(drop=True)
    return df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)


# ------------------------------------------------------------------------------
# Fingerprint
# ------------------------------------------------------------------------------
def compute_fingerprint(
    df_clean: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    max_samples: Optional[int],
    sampling_strategy: str,
    drop_cols: Optional[List[str]] = None,
    task_type: str = "",
) -> str:
    """Stable hash of dataset content + split configuration."""
    try:
        row_hashes = pd.util.hash_pandas_object(df_clean, index=False)
        content_hash = int(np.asarray(row_hashes, dtype=np.uint64).sum(dtype=np.uint64))
    except Exception:
        # Fallback for exotic dtypes: hash the CSV rendering.
        content_hash = int(
            hashlib.sha256(df_clean.to_csv(index=False).encode("utf-8")).hexdigest()[:16],
            16,
        )

    payload = json.dumps(
        {
            "shape": list(df_clean.shape),
            "columns": [str(c) for c in df_clean.columns],
            "content": str(content_hash),
            "target": str(target_col),
            "test_size": float(test_size),
            "random_state": int(random_state),
            "max_samples": int(max_samples) if max_samples else None,
            "sampling_strategy": sampling_strategy,
            "drop_cols": sorted(str(c) for c in (drop_cols or [])),
            "task_type": task_type,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_path(fingerprint: str) -> Path:
    return SPLITS_DIR / f"split_{fingerprint[:16]}.joblib"


# ------------------------------------------------------------------------------
# Partitioning
# ------------------------------------------------------------------------------
def _detect_task(y: pd.Series) -> str:
    # Imported lazily to avoid a module-level cycle with data_loader.
    from core.data_loader import detect_task_type

    return detect_task_type(y)


def _partition(
    df_clean: pd.DataFrame,
    target_col: str,
    task_type: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])

    stratify = None
    if task_type == TASK_CLASSIFICATION:
        class_counts = y.value_counts()
        if (class_counts >= 2).all() and len(class_counts) > 1:
            stratify = y

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def _write_split(
    fingerprint: str,
    partitions: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    metadata: Dict[str, Any],
) -> Path:
    path = _split_path(fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = partitions
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "metadata": metadata,
        },
        path,
        compress=3,
    )
    return path


def _read_split(path: Union[str, Path]) -> Dict[str, Any]:
    payload = joblib.load(path)
    required = {"X_train", "X_test", "y_train", "y_test"}
    if not required.issubset(payload):
        raise ValueError(f"Split file is malformed: {path}")
    return payload


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------
def split_or_create(
    df: pd.DataFrame,
    target_col: str,
    task: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_cols: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    sampling_strategy: str = SAMPLING_RANDOM,
    dataset_name: str = "Dataset",
    split_id: Optional[int] = None,
    save: bool = True,
    allow_reuse: bool = True,
) -> SplitResult:
    """Load a saved split, reuse an identical one, or create and persist a new one.

    ``split_id`` forces an explicit saved split (used by the UI). Otherwise a
    fingerprint of the cleaned content is matched when ``allow_reuse`` is true.
    """
    # 1. Explicit split selection wins.
    if split_id is not None:
        record = get_split(int(split_id))
        if record is None:
            raise ValueError(f"Saved split #{split_id} was not found.")
        path = Path(record["file_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Split file for #{split_id} is missing: {path}. Delete the record "
                "or create a new split."
            )
        payload = _read_split(path)
        return _result_from_payload(payload, record, source="loaded")

    # 2. Clean + subsample, then identify.
    df_clean = clean_dataframe(df, target_col, drop_cols)
    df_clean = subsample(df_clean, max_samples, random_state, sampling_strategy)
    y = df_clean[target_col]
    task_type = task if task in (TASK_CLASSIFICATION, TASK_REGRESSION) else _detect_task(y)

    fingerprint = compute_fingerprint(
        df_clean,
        target_col,
        test_size,
        random_state,
        max_samples,
        sampling_strategy,
        drop_cols,
        task_type,
    )

    # 3. Reuse an identical split when allowed.
    if allow_reuse:
        record = find_split_by_fingerprint(fingerprint)
        if record:
            path = Path(record["file_path"])
            if path.exists():
                payload = _read_split(path)
                return _result_from_payload(payload, record, source="reused")

    # 4. Create a fresh split.
    partitions = _partition(df_clean, target_col, task_type, test_size, random_state)
    classes = _classes_from_partitions(partitions, task_type)
    metadata = {
        "dataset_name": dataset_name,
        "target_col": target_col,
        "task_type": task_type,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "max_samples": int(max_samples) if max_samples else None,
        "sampling_strategy": sampling_strategy,
        "drop_cols": list(drop_cols or []),
        "classes": classes,
        "n_rows": len(df_clean),
    }

    new_split_id: Optional[int] = None
    if save:
        path = _write_split(fingerprint, partitions, metadata)
        metadata["file"] = str(path)
        new_split_id = save_split_record(
            fingerprint=fingerprint,
            dataset_name=dataset_name,
            target_col=target_col,
            task_type=task_type,
            test_size=test_size,
            random_state=random_state,
            max_samples=max_samples,
            sampling_strategy=sampling_strategy,
            drop_cols=list(drop_cols or []),
            n_rows=len(df_clean),
            n_train=len(partitions[0]),
            n_test=len(partitions[1]),
            feature_names=[str(c) for c in partitions[0].columns],
            file_path=str(path),
            config=metadata,
        )

    X_train, X_test, y_train, y_test = partitions
    return SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task_type=task_type,
        classes=classes,
        split_id=new_split_id,
        fingerprint=fingerprint,
        created=True,
        reused=False,
        n_rows=len(df_clean),
        n_train=len(X_train),
        n_test=len(X_test),
        config=metadata,
        source="created",
        clean_frame=df_clean,
    )


def _classes_from_partitions(
    partitions: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    task_type: str,
) -> List[Any]:
    if task_type != TASK_CLASSIFICATION:
        return []
    _, _, y_train, y_test = partitions
    values = pd.concat([y_train, y_test]).unique().tolist()
    return sorted(values, key=lambda value: str(value))


def _result_from_payload(
    payload: Dict[str, Any], record: Dict[str, Any], source: str
) -> SplitResult:
    X_train = payload["X_train"]
    X_test = payload["X_test"]
    y_train = payload["y_train"]
    y_test = payload["y_test"]
    task_type = record.get("task_type") or (payload.get("metadata") or {}).get(
        "task_type", TASK_CLASSIFICATION
    )
    classes = (payload.get("metadata") or {}).get("classes")
    if classes is None:
        classes = _classes_from_partitions((X_train, X_test, y_train, y_test), task_type)

    target_col = record.get("target_col")
    clean_frame = _reconstruct_source(X_train, y_train, X_test, y_test, target_col)

    return SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task_type=task_type,
        classes=list(classes),
        split_id=int(record["id"]),
        fingerprint=record.get("fingerprint", ""),
        created=False,
        reused=source == "reused",
        n_rows=record.get("n_rows") or (len(X_train) + len(X_test)),
        n_train=len(X_train),
        n_test=len(X_test),
        config=record.get("config") or {},
        source=source,
        clean_frame=clean_frame,
    )


def _reconstruct_source(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_col: Optional[str],
) -> Optional[pd.DataFrame]:
    """Rebuild the cleaned source frame from the two partitions.

    The split preserves the cleaned frame's RangeIndex, so sorting by index
    restores the original row order (needed for time-series noise profiles).
    """
    if not target_col:
        return None
    try:
        train = X_train.copy()
        train[target_col] = y_train
        test = X_test.copy()
        test[target_col] = y_test
        combined = pd.concat([train, test])
        if combined.index.is_unique:
            combined = combined.sort_index()
        return combined.reset_index(drop=True)
    except Exception:
        return None


def preview_fingerprint(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    drop_cols: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    sampling_strategy: str = SAMPLING_RANDOM,
    task: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Look up the saved split a run would reuse, without creating anything.

    Returns the split record (including its id) or ``None``. Used by the UI to
    tell the user in advance whether a run will reuse or create a split.
    """
    try:
        df_clean = clean_dataframe(df, target_col, drop_cols)
        df_clean = subsample(df_clean, max_samples, random_state, sampling_strategy)
        y = df_clean[target_col]
        task_type = task if task in (TASK_CLASSIFICATION, TASK_REGRESSION) else _detect_task(y)
        fingerprint = compute_fingerprint(
            df_clean,
            target_col,
            test_size,
            random_state,
            max_samples,
            sampling_strategy,
            drop_cols,
            task_type,
        )
        record = find_split_by_fingerprint(fingerprint)
        if record and Path(record["file_path"]).exists():
            return record
        return None
    except Exception:
        return None
