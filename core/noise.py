"""
Noise treatment, outlier handling and optional synthetic noise injection.

The module is deliberately *row-local* and non-parametric where possible, so it
can be applied to the training and test partitions independently without leaking
information across the split. Fitted quantities (clip fences, Isolation Forest)
are learned on the training partition only.

Every configuration exposes ``to_dict()`` / ``from_dict()`` so runs can be
persisted in SQLite and reconstructed for the dashboard and the AI context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # scipy ships with scikit-learn's usual environment, but stay defensive
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - optional
    savgol_filter = None

from config import NOISE_STRATEGIES, OUTLIER_METHODS


# ------------------------------------------------------------------------------
# Configuration objects
# ------------------------------------------------------------------------------
@dataclass
class NoiseConfig:
    strategy: str = "none"
    columns: List[str] = field(default_factory=list)  # empty = every numeric column
    window: int = 5
    center: bool = False
    polyorder: int = 2
    k: float = 3.0
    order_column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "NoiseConfig":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def enabled(self) -> bool:
        return self.strategy in NOISE_STRATEGIES and self.strategy != "none"


@dataclass
class OutlierConfig:
    method: str = "none"
    action: str = "clip"
    columns: List[str] = field(default_factory=list)
    k: float = 1.5
    contamination: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OutlierConfig":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def enabled(self) -> bool:
        return self.method in OUTLIER_METHODS and self.method != "none"


@dataclass
class InjectionConfig:
    method: str = "none"
    columns: List[str] = field(default_factory=list)
    ratio: float = 0.05
    spike_ratio: float = 0.01
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "InjectionConfig":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def enabled(self) -> bool:
        return self.method in ("gaussian", "spikes", "missing")


def treatment_enabled(noise: NoiseConfig, outliers: OutlierConfig) -> bool:
    return bool(noise.enabled or outliers.enabled)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def numeric_columns(df: pd.DataFrame, requested: Optional[List[str]] = None) -> List[str]:
    """Numeric columns to treat: the requested subset, else all numeric columns."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not requested:
        return numeric
    return [c for c in requested if c in numeric]


def _odd_window(window: int, polyorder: int) -> int:
    value = max(3, int(window))
    if value % 2 == 0:
        value += 1
    return max(value, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)


def _order_and_inverse(df: pd.DataFrame, order_column: Optional[str]) -> np.ndarray:
    """Indices that restore the dataset's original order before smoothing."""
    if order_column and order_column in df.columns:
        keys = df[order_column].to_numpy()
    elif df.index.is_monotonic_increasing:
        keys = df.index.to_numpy()
    else:
        return np.arange(len(df))
    try:
        return np.argsort(keys, kind="mergesort")
    except TypeError:
        return np.arange(len(df))


def _restore_order(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    restored = np.empty_like(values)
    restored[order] = values
    return restored


def _smooth_series(values: pd.Series, config: NoiseConfig) -> pd.Series:
    """Apply one smoothing strategy to a Series already in the target order."""
    strategy = config.strategy
    window = max(1, int(config.window))

    if strategy == "rolling_mean":
        return values.rolling(window=window, min_periods=1, center=config.center).mean()
    if strategy == "rolling_median":
        return values.rolling(window=window, min_periods=1, center=config.center).median()
    if strategy == "ema":
        return values.ewm(span=window, adjust=False).mean()
    if strategy == "interpolate":
        return values.interpolate(method="linear", limit_direction="both")
    if strategy == "ffill":
        return values.ffill()
    if strategy == "hampel":
        median = values.rolling(window=window, min_periods=1, center=True).median()
        mad = (values - median).abs().rolling(window=window, min_periods=1, center=True).median()
        threshold = float(config.k) * 1.4826 * mad
        flagged = (values - median).abs() > threshold
        out = values.copy()
        out[flagged] = median[flagged]
        return out
    if strategy == "savgol":
        if savgol_filter is None:
            return values.rolling(window=window, min_periods=1, center=True).mean()
        length = _odd_window(window, int(config.polyorder))
        length = min(length, len(values) if len(values) % 2 == 1 else max(3, len(values) - 1))
        polyorder = min(int(config.polyorder), max(1, length - 2))
        if length < 3 or length > len(values):
            return values.rolling(window=window, min_periods=1, center=True).mean()
        series = values.interpolate(method="linear", limit_direction="both")
        if series.isna().any():
            return values
        smoothed = savgol_filter(
            series.to_numpy(dtype=float),
            window_length=length,
            polyorder=polyorder,
        )
        return pd.Series(smoothed, index=values.index)
    if strategy in ("iqr_clip", "zscore_clip"):
        return values
    return values


# ------------------------------------------------------------------------------
# Noise treatment
# ------------------------------------------------------------------------------
class NoiseTreater:
    """Apply a configured denoising strategy to selected numeric columns."""

    def __init__(self, config: Optional[NoiseConfig] = None):
        self.config = config or NoiseConfig()
        self.columns: List[str] = []
        self.bounds: Dict[str, Tuple[float, float]] = {}
        self.fallback: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame) -> "NoiseTreater":
        self.columns = numeric_columns(X_train, self.config.columns)
        for column in self.columns:
            series = pd.to_numeric(X_train[column], errors="coerce")
            self.fallback[column] = float(series.median()) if series.notna().any() else 0.0
            if self.config.strategy == "iqr_clip":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                self.bounds[column] = (
                    float(q1 - self.config.k * iqr),
                    float(q3 + self.config.k * iqr),
                )
            elif self.config.strategy == "zscore_clip":
                mean, std = series.mean(), series.std()
                std = float(std) if np.isfinite(std) and std > 0 else 0.0
                self.bounds[column] = (
                    float(mean - self.config.k * std),
                    float(mean + self.config.k * std),
                )
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("NoiseTreater must be fitted before transform().")
        if not self.config.enabled or not self.columns:
            return X

        out = X.copy()
        order = _order_and_inverse(X, self.config.order_column)

        for column in self.columns:
            if column not in out.columns:
                continue
            series = pd.to_numeric(out[column], errors="coerce")
            mask = series.isna().to_numpy()
            sorted_series = pd.Series(
                series.to_numpy()[order], index=np.arange(len(order)), dtype=float
            )

            if self.config.strategy in ("iqr_clip", "zscore_clip"):
                low, high = self.bounds.get(column, (-np.inf, np.inf))
                filled = sorted_series.median() if sorted_series.notna().any() else 0.0
                clipped = sorted_series.fillna(filled).clip(lower=low, upper=high)
                smoothed = clipped
            else:
                smoothed = _smooth_series(sorted_series, self.config)

            values = _restore_order(
                np.asarray(smoothed, dtype=float), order
            )
            if mask.any():
                # Keep missing values missing so imputation still sees them.
                values[mask] = np.nan
            out[column] = values

        return out

    def describe(self) -> Dict[str, Any]:
        return {
            "strategy": self.config.strategy,
            "columns": list(self.columns),
            "window": self.config.window,
            "center": self.config.center,
            "polyorder": self.config.polyorder,
            "k": self.config.k,
            "order_column": self.config.order_column,
            "enabled": self.config.enabled,
        }


# ------------------------------------------------------------------------------
# Outlier handling
# ------------------------------------------------------------------------------
class OutlierTreater:
    """Clip or flag outliers in numeric columns."""

    def __init__(self, config: Optional[OutlierConfig] = None, random_state: int = 42):
        self.config = config or OutlierConfig()
        self.columns: List[str] = []
        self.bounds: Dict[str, Tuple[float, float]] = {}
        self.medians: Dict[str, float] = {}
        self.model = None
        self.is_fitted = False
        self.random_state = random_state
        self.removed_rows = 0

    def fit(self, X_train: pd.DataFrame) -> "OutlierTreater":
        self.columns = numeric_columns(X_train, self.config.columns)
        if not self.config.enabled or not self.columns:
            self.is_fitted = True
            return self

        for column in self.columns:
            series = pd.to_numeric(X_train[column], errors="coerce")
            self.medians[column] = (
                float(series.median()) if series.notna().any() else 0.0
            )
            if self.config.method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                self.bounds[column] = (
                    float(q1 - self.config.k * iqr),
                    float(q3 + self.config.k * iqr),
                )
            elif self.config.method == "zscore":
                mean = float(series.mean()) if series.notna().any() else 0.0
                std = float(series.std()) if series.notna().any() else 0.0
                std = std if np.isfinite(std) and std > 0 else 0.0
                self.bounds[column] = (
                    mean - self.config.k * std,
                    mean + self.config.k * std,
                )

        if self.config.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            matrix = self._matrix(X_train)
            self.model = IsolationForest(
                contamination=self.config.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            ).fit(matrix)

        self.is_fitted = True
        return self

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        frame = X[self.columns].apply(pd.to_numeric, errors="coerce")
        medians = pd.Series(self.medians)
        frame = frame.fillna(medians.reindex(frame.columns))
        return frame.to_numpy(dtype=float)

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Return the treated frame plus a boolean mask of removed rows."""
        if not self.is_fitted:
            raise RuntimeError("OutlierTreater must be fitted before transform().")
        removed = np.zeros(len(X), dtype=bool)
        if not self.config.enabled or not self.columns:
            return X, removed

        out = X.copy()

        if self.config.method in ("iqr", "zscore"):
            for column in self.columns:
                if column not in out.columns:
                    continue
                series = pd.to_numeric(out[column], errors="coerce")
                low, high = self.bounds.get(column, (-np.inf, np.inf))
                if self.config.action == "remove":
                    removed |= ((series < low) | (series > high)).to_numpy()
                else:
                    out[column] = series.clip(lower=low, upper=high)

        elif self.config.method == "isolation_forest" and self.model is not None:
            flags = self.model.predict(self._matrix(out)) == -1
            if self.config.action == "remove":
                removed |= flags
            else:
                # The forest has no bounds: neutralise flagged cells so the
                # imputer can treat them as missing.
                for column in self.columns:
                    if column in out.columns:
                        out.loc[flags, column] = np.nan

        self.removed_rows = int(removed.sum())
        return out, removed

    def describe(self) -> Dict[str, Any]:
        return {
            "method": self.config.method,
            "action": self.config.action,
            "columns": list(self.columns),
            "k": self.config.k,
            "contamination": self.config.contamination,
            "removed_rows": self.removed_rows,
            "enabled": self.config.enabled,
        }


# ------------------------------------------------------------------------------
# Synthetic noise injection
# ------------------------------------------------------------------------------
def inject_noise(
    df: pd.DataFrame,
    config: Optional[InjectionConfig] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add controlled synthetic noise so a clean dataset still has a noisy side.

    Returns the modified frame and a report suitable for the data-quality JSON.
    Injection is deterministic for a fixed ``seed``.
    """
    config = config or InjectionConfig()
    report: Dict[str, Any] = {
        "method": config.method,
        "enabled": config.enabled,
        "columns": [],
        "ratio": config.ratio,
        "spike_ratio": config.spike_ratio,
        "seed": int(random_state),
        "cells_changed": 0,
    }
    if not config.enabled or df.empty:
        return df, report

    columns = numeric_columns(df, config.columns)
    if not columns:
        return df, report

    out = df.copy()
    rng = np.random.default_rng(int(config.seed if config.seed is not None else random_state))
    changed = 0

    if config.method == "gaussian":
        for column in columns:
            series = pd.to_numeric(out[column], errors="coerce")
            std = float(series.std())
            if not np.isfinite(std) or std <= 0:
                continue
            noise = rng.normal(0.0, config.ratio * std, size=len(out))
            out[column] = series + noise
            changed += int(np.isfinite(noise).sum())

    elif config.method == "spikes":
        for column in columns:
            series = pd.to_numeric(out[column], errors="coerce")
            std = float(series.std())
            if not np.isfinite(std) or std <= 0:
                continue
            n_spikes = max(1, int(config.spike_ratio * len(out)))
            positions = rng.choice(len(out), size=min(n_spikes, len(out)), replace=False)
            signs = rng.choice([-1.0, 1.0], size=len(positions))
            magnitudes = signs * float(config.ratio) * std
            values = series.to_numpy(dtype=float).copy()
            values[positions] = values[positions] + magnitudes
            out[column] = values
            changed += len(positions)

    elif config.method == "missing":
        for column in columns:
            n_missing = int(config.ratio * len(out))
            if n_missing <= 0:
                continue
            positions = rng.choice(len(out), size=min(n_missing, len(out)), replace=False)
            values = pd.to_numeric(out[column], errors="coerce").to_numpy(
                dtype=float, copy=True
            )
            values[positions] = np.nan
            out[column] = values
            changed += len(positions)

    report["columns"] = columns
    report["cells_changed"] = changed
    return out, report


# ------------------------------------------------------------------------------
# Convenience: build treaters from serialized configs
# ------------------------------------------------------------------------------
def build_treaters(
    noise: NoiseConfig,
    outliers: OutlierConfig,
    X_train: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[NoiseTreater, OutlierTreater]:
    treater = NoiseTreater(noise).fit(X_train)
    outlier = OutlierTreater(outliers, random_state=random_state).fit(X_train)
    return treater, outlier
