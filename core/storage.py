"""
SQLite Persistence Layer for ML2 Training Pipeline.
Stores experiment metadata, model benchmarks, evaluation metrics, and predictions.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import DB_PATH

_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db() -> None:
    """Initialize SQLite database tables if not already present."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name            TEXT NOT NULL,
                    dataset_name        TEXT NOT NULL,
                    task_type           TEXT NOT NULL,
                    target_col          TEXT NOT NULL,
                    n_train             INTEGER,
                    n_test              INTEGER,
                    n_features          INTEGER,
                    feature_names_json  TEXT,
                    created_at          TEXT NOT NULL,
                    status              TEXT DEFAULT 'completed'
                );

                CREATE TABLE IF NOT EXISTS model_results (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id           INTEGER NOT NULL,
                    model_id                TEXT NOT NULL,
                    model_name              TEXT NOT NULL,
                    category                TEXT NOT NULL,
                    train_time_sec          REAL,
                    metrics_json            TEXT NOT NULL,
                    params_json             TEXT,
                    feature_importances_json TEXT,
                    predictions_sample_json  TEXT,
                    created_at              TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_results_exp_id ON model_results(experiment_id);
                """
            )
            conn.commit()
            _INITIALIZED = True
        finally:
            conn.close()


def save_experiment(
    run_name: str,
    dataset_name: str,
    task_type: str,
    target_col: str,
    n_train: int,
    n_test: int,
    n_features: int,
    feature_names: Optional[List[str]] = None,
    status: str = "completed",
) -> int:
    """Insert a new experiment record and return its primary key ID."""
    init_db()
    conn = _connect()
    try:
        now = datetime.now().isoformat(timespec="seconds")
        cursor = conn.execute(
            """
            INSERT INTO experiments (
                run_name, dataset_name, task_type, target_col,
                n_train, n_test, n_features, feature_names_json, created_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                dataset_name,
                task_type,
                target_col,
                n_train,
                n_test,
                n_features,
                json.dumps(feature_names or []),
                now,
                status,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def save_model_result(
    experiment_id: int,
    model_id: str,
    model_name: str,
    category: str,
    train_time_sec: float,
    metrics: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    feature_importances: Optional[Dict[str, float]] = None,
    predictions_sample: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert a model benchmark result under an experiment."""
    init_db()
    conn = _connect()
    try:
        now = datetime.now().isoformat(timespec="seconds")
        cursor = conn.execute(
            """
            INSERT INTO model_results (
                experiment_id, model_id, model_name, category, train_time_sec,
                metrics_json, params_json, feature_importances_json,
                predictions_sample_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                model_id,
                model_name,
                category,
                round(float(train_time_sec), 4),
                json.dumps(metrics),
                json.dumps(params or {}),
                json.dumps(feature_importances or {}),
                json.dumps(predictions_sample or {}),
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_experiments() -> List[Dict[str, Any]]:
    """Retrieve all recorded experiments ordered by creation time descending."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT e.*, COUNT(r.id) as model_count
            FROM experiments e
            LEFT JOIN model_results r ON e.id = r.experiment_id
            GROUP BY e.id
            ORDER BY e.id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_experiment(experiment_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve details for a single experiment."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_leaderboard(experiment_id: int) -> List[Dict[str, Any]]:
    """Retrieve model results for an experiment with parsed JSON fields."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT * FROM model_results
            WHERE experiment_id = ?
            ORDER BY id ASC
            """,
            (experiment_id,),
        ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d.get("metrics_json") or "{}")
            d["params"] = json.loads(d.get("params_json") or "{}")
            d["feature_importances"] = json.loads(
                d.get("feature_importances_json") or "{}"
            )
            d["predictions_sample"] = json.loads(
                d.get("predictions_sample_json") or "{}"
            )
            results.append(d)
        return results
    finally:
        conn.close()


def delete_experiment(experiment_id: int) -> bool:
    """Delete an experiment and all associated model results."""
    init_db()
    conn = _connect()
    try:
        conn.execute("DELETE FROM model_results WHERE experiment_id = ?", (experiment_id,))
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def export_leaderboard_df(experiment_id: int) -> pd.DataFrame:
    """Format leaderboard into a clean tabular pandas DataFrame."""
    results = get_leaderboard(experiment_id)
    if not results:
        return pd.DataFrame()

    records = []
    for r in results:
        row_dict = {
            "Model": r["model_name"],
            "Category": r["category"].capitalize(),
            "Training Time (s)": r["train_time_sec"],
        }
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)):
                row_dict[k.upper()] = v
        records.append(row_dict)

    return pd.DataFrame(records)

