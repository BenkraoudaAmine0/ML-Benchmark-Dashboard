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

from config import DB_PATH, TASK_CLASSIFICATION

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


def _ensure_columns(
    conn: sqlite3.Connection, table: str, columns: Dict[str, str]
) -> None:
    """Add any missing columns to an existing table (in-place migration)."""
    existing = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


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

                CREATE TABLE IF NOT EXISTS dataset_splits (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint         TEXT NOT NULL UNIQUE,
                    dataset_name        TEXT NOT NULL,
                    target_col          TEXT NOT NULL,
                    task_type           TEXT NOT NULL,
                    test_size           REAL,
                    random_state        INTEGER,
                    max_samples         INTEGER,
                    sampling_strategy   TEXT,
                    drop_cols_json      TEXT,
                    n_rows              INTEGER,
                    n_train             INTEGER,
                    n_test              INTEGER,
                    feature_names_json  TEXT,
                    file_path           TEXT NOT NULL,
                    config_json         TEXT,
                    created_at          TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ai_insights (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   INTEGER NOT NULL,
                    prompt_version  TEXT NOT NULL,
                    provider        TEXT,
                    model           TEXT,
                    context_hash    TEXT,
                    content_json    TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_ai_exp ON ai_insights(experiment_id);
                """
            )

            # Incremental migrations for databases created before these features.
            _ensure_columns(
                conn,
                "experiments",
                {
                    "split_id": "INTEGER",
                    "comparison_mode": "INTEGER DEFAULT 0",
                    "noise_config_json": "TEXT",
                    "imputer_config_json": "TEXT",
                    "data_quality_json": "TEXT",
                },
            )
            _ensure_columns(
                conn,
                "model_results",
                {"variant": "TEXT NOT NULL DEFAULT 'baseline'"},
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_results_exp_variant "
                "ON model_results(experiment_id, variant)"
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
    *,
    split_id: Optional[int] = None,
    comparison_mode: bool = False,
    noise_config: Optional[Dict[str, Any]] = None,
    imputer_config: Optional[Dict[str, Any]] = None,
    data_quality: Optional[Dict[str, Any]] = None,
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
                n_train, n_test, n_features, feature_names_json, created_at, status,
                split_id, comparison_mode, noise_config_json, imputer_config_json,
                data_quality_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                split_id,
                1 if comparison_mode else 0,
                json.dumps(noise_config or {}),
                json.dumps(imputer_config or {}),
                json.dumps(data_quality or {}),
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
    variant: str = "baseline",
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
                predictions_sample_json, created_at, variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                variant or "baseline",
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


def get_benchmark_history() -> List[Dict[str, Any]]:
    """Best primary score per experiment, chronological, for the history trend chart."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT e.id, e.run_name, e.dataset_name, e.task_type, e.created_at,
                   r.model_name, r.metrics_json, r.variant
            FROM experiments e
            JOIN model_results r ON r.experiment_id = e.id
            ORDER BY e.id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    # Comparison experiments store both variants. The trend chart is about
    # experiment progress, not treatment, so it reads the baseline variant when
    # one exists (falling back to whatever variant the run recorded).
    ids_with_baseline = {
        row["id"] for row in rows if row["variant"] in (None, "", "baseline")
    }

    history: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if row["id"] in ids_with_baseline and row["variant"] not in (None, "", "baseline"):
            continue
        record = history.setdefault(
            row["id"],
            {
                "id": row["id"],
                "run_name": row["run_name"],
                "dataset_name": row["dataset_name"],
                "task_type": row["task_type"],
                "created_at": row["created_at"],
                "best_score": None,
                "best_model": "",
                "mean_score": None,
                "model_count": 0,
            },
        )
        primary_key = "f1_macro" if row["task_type"] == TASK_CLASSIFICATION else "r2"
        metrics = json.loads(row["metrics_json"] or "{}")
        score = metrics.get(primary_key)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            record.setdefault("_scores", []).append(float(score))
            if record["best_score"] is None or score > record["best_score"]:
                record["best_score"] = float(score)
                record["best_model"] = row["model_name"]
        record["model_count"] += 1

    for record in history.values():
        scores = record.pop("_scores", [])
        if scores:
            record["mean_score"] = round(sum(scores) / len(scores), 4)
    return list(history.values())


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


def get_leaderboard(
    experiment_id: int, variant: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve model results for an experiment with parsed JSON fields.

    ``variant=None`` returns every row (baseline and treated); pass a variant
    name to filter a comparison experiment.
    """
    init_db()
    conn = _connect()
    try:
        if variant is None:
            rows = conn.execute(
                """
                SELECT * FROM model_results
                WHERE experiment_id = ?
                ORDER BY id ASC
                """,
                (experiment_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM model_results
                WHERE experiment_id = ? AND variant = ?
                ORDER BY id ASC
                """,
                (experiment_id, variant),
            ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d["variant"] = d.get("variant") or "baseline"
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


def get_variants(experiment_id: int) -> List[str]:
    """Distinct variants recorded for an experiment, in insertion order."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT variant, MIN(id) AS first_id
            FROM model_results
            WHERE experiment_id = ?
            GROUP BY variant
            ORDER BY first_id ASC
            """,
            (experiment_id,),
        ).fetchall()
        return [row["variant"] or "baseline" for row in rows]
    finally:
        conn.close()


def default_variant(experiment_id: int) -> Optional[str]:
    """Pick the variant a dashboard should show by default.

    Baseline is the untreated reference, so it is preferred; single-variant
    experiments simply return the variant they recorded.
    """
    variants = get_variants(experiment_id)
    if not variants:
        return None
    return "baseline" if "baseline" in variants else variants[0]


def delete_experiment(experiment_id: int) -> bool:
    """Delete an experiment and all associated model results."""
    init_db()
    conn = _connect()
    try:
        conn.execute("DELETE FROM model_results WHERE experiment_id = ?", (experiment_id,))
        conn.execute("DELETE FROM ai_insights WHERE experiment_id = ?", (experiment_id,))
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def export_leaderboard_df(
    experiment_id: int, variant: Optional[str] = None
) -> pd.DataFrame:
    """Format leaderboard into a clean tabular pandas DataFrame."""
    results = get_leaderboard(experiment_id, variant=variant)
    if not results:
        return pd.DataFrame()

    include_variant = len({r["variant"] for r in results}) > 1
    records = []
    for r in results:
        row_dict = {
            "Model": r["model_name"],
            "Category": r["category"].capitalize(),
        }
        if include_variant:
            row_dict["Variant"] = r["variant"]
        row_dict["Training Time (s)"] = r["train_time_sec"]
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)):
                row_dict[k.upper()] = v
        records.append(row_dict)

    return pd.DataFrame(records)


# ------------------------------------------------------------------------------
# Persisted train/test splits
# ------------------------------------------------------------------------------
def save_split_record(
    fingerprint: str,
    dataset_name: str,
    target_col: str,
    task_type: str,
    test_size: float,
    random_state: int,
    max_samples: Optional[int],
    sampling_strategy: str,
    drop_cols: List[str],
    n_rows: int,
    n_train: int,
    n_test: int,
    feature_names: List[str],
    file_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert a split record (idempotent per fingerprint) and return its id."""
    init_db()
    conn = _connect()
    try:
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            """
            INSERT OR IGNORE INTO dataset_splits (
                fingerprint, dataset_name, target_col, task_type, test_size,
                random_state, max_samples, sampling_strategy, drop_cols_json,
                n_rows, n_train, n_test, feature_names_json, file_path,
                config_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fingerprint,
                dataset_name,
                target_col,
                task_type,
                float(test_size),
                int(random_state),
                int(max_samples) if max_samples else None,
                sampling_strategy,
                json.dumps(drop_cols or []),
                int(n_rows),
                int(n_train),
                int(n_test),
                json.dumps(feature_names or []),
                file_path,
                json.dumps(config or {}),
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM dataset_splits WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        return int(row["id"])
    finally:
        conn.close()


def get_split(split_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a saved split record by id."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM dataset_splits WHERE id = ?", (split_id,)
        ).fetchone()
        return _parse_split_row(row) if row else None
    finally:
        conn.close()


def find_split_by_fingerprint(fingerprint: str) -> Optional[Dict[str, Any]]:
    """Retrieve a saved split record by its content fingerprint."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM dataset_splits WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        return _parse_split_row(row) if row else None
    finally:
        conn.close()


def list_splits(
    dataset_name: Optional[str] = None, target_col: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List saved splits, newest first, optionally filtered by dataset/target."""
    init_db()
    conn = _connect()
    try:
        clause = []
        params: List[Any] = []
        if dataset_name:
            clause.append("dataset_name = ?")
            params.append(dataset_name)
        if target_col:
            clause.append("target_col = ?")
            params.append(target_col)
        where = f"WHERE {' AND '.join(clause)}" if clause else ""
        rows = conn.execute(
            f"SELECT * FROM dataset_splits {where} ORDER BY id DESC", params
        ).fetchall()
        return [_parse_split_row(row) for row in rows]
    finally:
        conn.close()


def delete_split(split_id: int) -> bool:
    """Delete a split record and its joblib file."""
    init_db()
    record = get_split(split_id)
    if not record:
        return False
    conn = _connect()
    try:
        conn.execute("DELETE FROM dataset_splits WHERE id = ?", (split_id,))
        conn.commit()
    finally:
        conn.close()
    path = Path(record.get("file_path") or "")
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    return True


def _parse_split_row(row: sqlite3.Row) -> Dict[str, Any]:
    record = dict(row)
    record["drop_cols"] = json.loads(record.get("drop_cols_json") or "[]")
    record["feature_names"] = json.loads(record.get("feature_names_json") or "[]")
    record["config"] = json.loads(record.get("config_json") or "{}")
    return record


# ------------------------------------------------------------------------------
# AI interpretation cache
# ------------------------------------------------------------------------------
def save_ai_insight(
    experiment_id: int,
    prompt_version: str,
    provider: str,
    model: str,
    context_hash: str,
    content: Dict[str, Any],
) -> int:
    """Store (or replace) the AI report for an experiment/model combination."""
    init_db()
    conn = _connect()
    try:
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "DELETE FROM ai_insights WHERE experiment_id = ? AND prompt_version = ? AND model = ?",
            (experiment_id, prompt_version, model),
        )
        cursor = conn.execute(
            """
            INSERT INTO ai_insights (
                experiment_id, prompt_version, provider, model, context_hash,
                content_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                prompt_version,
                provider,
                model,
                context_hash,
                json.dumps(content),
                now,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def get_ai_insight(
    experiment_id: int,
    prompt_version: str,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve the newest cached AI report for an experiment."""
    init_db()
    conn = _connect()
    try:
        if model:
            row = conn.execute(
                """
                SELECT * FROM ai_insights
                WHERE experiment_id = ? AND prompt_version = ? AND model = ?
                ORDER BY id DESC LIMIT 1
                """,
                (experiment_id, prompt_version, model),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM ai_insights
                WHERE experiment_id = ? AND prompt_version = ?
                ORDER BY id DESC LIMIT 1
                """,
                (experiment_id, prompt_version),
            ).fetchone()
        if not row:
            return None
        record = dict(row)
        record["content"] = json.loads(record.get("content_json") or "{}")
        return record
    finally:
        conn.close()


def delete_ai_insight(experiment_id: int, prompt_version: Optional[str] = None) -> int:
    """Remove cached AI reports for an experiment; returns rows deleted."""
    init_db()
    conn = _connect()
    try:
        if prompt_version:
            cursor = conn.execute(
                "DELETE FROM ai_insights WHERE experiment_id = ? AND prompt_version = ?",
                (experiment_id, prompt_version),
            )
        else:
            cursor = conn.execute(
                "DELETE FROM ai_insights WHERE experiment_id = ?", (experiment_id,)
            )
        conn.commit()
        return int(cursor.rowcount or 0)
    finally:
        conn.close()

