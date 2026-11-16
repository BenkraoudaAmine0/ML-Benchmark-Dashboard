"""
Training Pipeline Orchestrator.
Coordinates data loading, persistent splitting, preprocessing, optional noise /
outlier treatment, model training, evaluation, and SQLite persistence.

A run can carry one or two *variants* of the same split:
``baseline`` (untreated, optionally with synthetic noise injected) and
``treated`` (denoised / outlier-handled). Training both in one experiment keeps
the paired comparison exact: identical rows, identical test set.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Union

# Suppress sklearn future warnings during fitting
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from config import (
    ARTIFACTS_DIR,
    CAT_CLASSICAL,
    CAT_ENSEMBLE,
    NAN_CAPABLE_MODELS,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    VARIANT_BASELINE,
    VARIANT_TREATED,
)
from core.data_quality import (
    attach_histograms,
    compare_variants,
    primary_metric_for,
    profile_columns,
    profile_dataset,
)
from core.metrics import evaluate_model
from core.noise import (
    InjectionConfig,
    NoiseConfig,
    NoiseTreater,
    OutlierConfig,
    OutlierTreater,
    inject_noise,
    numeric_columns,
    treatment_enabled,
)
from core.preprocessor import DataPreprocessor
from core.splits import SAMPLING_RANDOM, SplitResult, split_or_create
from core.storage import save_experiment, save_model_result
from models.registry import MODEL_REGISTRY, instantiate_trainer


@dataclass
class VariantData:
    """One processed view of the split."""

    name: str
    preprocessor: DataPreprocessor
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    rows_removed: int = 0
    # Fitted treatment steps (None on the baseline variant) so live inference
    # can apply the same clipping / smoothing to a single sample.
    noise_treater: Optional[NoiseTreater] = None
    outlier_treater: Optional[OutlierTreater] = None


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
        *,
        dataset_name: str = "Dataset",
        sampling_strategy: str = SAMPLING_RANDOM,
        numeric_imputer: str = "median",
        categorical_imputer: str = "most_frequent",
        imputer_params: Optional[Dict[str, Any]] = None,
        noise_config: Optional[Union[NoiseConfig, Dict[str, Any]]] = None,
        outlier_config: Optional[Union[OutlierConfig, Dict[str, Any]]] = None,
        injection_config: Optional[Union[InjectionConfig, Dict[str, Any]]] = None,
        compare_noise: bool = False,
        split_id: Optional[int] = None,
        save_split: bool = True,
        allow_split_reuse: bool = True,
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
        self.dataset_name = dataset_name
        self.sampling_strategy = sampling_strategy

        self.numeric_imputer = numeric_imputer
        self.categorical_imputer = categorical_imputer
        self.imputer_params = dict(imputer_params or {})

        self.noise_config = _coerce_noise_config(noise_config)
        self.outlier_config = _coerce_outlier_config(outlier_config)
        self.injection_config = _coerce_injection_config(injection_config)
        self.compare_noise = bool(compare_noise)

        self.split_id = split_id
        self.save_split = save_split
        self.allow_split_reuse = allow_split_reuse

        # Prepared data components
        self.split: Optional[SplitResult] = None
        self.X_train_raw: Optional[pd.DataFrame] = None
        self.X_test_raw: Optional[pd.DataFrame] = None
        self.y_train_raw: Optional[pd.Series] = None
        self.y_test_raw: Optional[pd.Series] = None
        self.classes_list: List[Any] = []
        self.task_type: str = ""

        # Variants: name -> processed view
        self.variants: Dict[str, VariantData] = {}
        self.data_quality: Dict[str, Any] = {}
        self.injection_report: Dict[str, Any] = {}
        self.noise_report: Dict[str, Any] = {}
        self.outlier_report: Dict[str, Any] = {}

        # Backwards-compatible aliases (first variant)
        self.preprocessor = DataPreprocessor(
            scale_numeric=scale_numeric,
            numeric_imputer=numeric_imputer,
            categorical_imputer=categorical_imputer,
            imputer_params=self.imputer_params,
        )
        self.X_train_proc: Optional[np.ndarray] = None
        self.X_test_proc: Optional[np.ndarray] = None
        self.y_train_proc: Optional[np.ndarray] = None
        self.y_test_proc: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

        # Fitted trainers: variant -> model_id -> trainer
        self.fitted_trainers: Dict[str, Dict[str, Any]] = {}
        self._last_trainer: Any = None

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    @property
    def comparison_mode(self) -> bool:
        return len(self.variants) > 1

    @property
    def variant_names(self) -> List[str]:
        return list(self.variants.keys())

    @property
    def treatment_summary(self) -> Dict[str, Any]:
        return {
            "noise": self.noise_report,
            "outliers": self.outlier_report,
            "injection": self.injection_report,
            "compare": self.comparison_mode,
        }

    def prepare(self) -> None:
        """Split, optionally inject / treat, and preprocess every variant."""
        split = split_or_create(
            df=self.df,
            target_col=self.target_col,
            task=self.task,
            test_size=self.test_size,
            random_state=self.random_state,
            drop_cols=self.drop_cols,
            max_samples=self.max_samples,
            sampling_strategy=self.sampling_strategy,
            dataset_name=self.dataset_name,
            split_id=self.split_id,
            save=self.save_split,
            allow_reuse=self.allow_split_reuse,
        )
        self.split = split
        self.task_type = split.task_type
        self.classes_list = split.classes
        self.X_train_raw = split.X_train
        self.X_test_raw = split.X_test
        self.y_train_raw = split.y_train
        self.y_test_raw = split.y_test

        # 1. Optional synthetic noise injection (the "noisy" side).
        baseline_train = split.X_train
        baseline_test = split.X_test
        self.injection_report = {"enabled": False, "method": "none"}
        if self.injection_config.enabled:
            # Different deterministic streams for train and test so the same
            # cells are not corrupted on both sides.
            baseline_train, train_report = inject_noise(
                split.X_train,
                replace(self.injection_config, seed=self.random_state),
                random_state=self.random_state,
            )
            baseline_test, test_report = inject_noise(
                split.X_test,
                replace(self.injection_config, seed=self.random_state + 1),
                random_state=self.random_state + 1,
            )
            self.injection_report = {
                **train_report,
                "cells_changed": int(train_report.get("cells_changed", 0))
                + int(test_report.get("cells_changed", 0)),
                "enabled": True,
            }

        # 2. Treatment view (denoise + outlier handling), fitted on train only.
        treated_train = baseline_train
        treated_test = baseline_test
        treated_y_train = split.y_train
        removed_total = 0
        self.noise_report = {"enabled": False, "strategy": "none"}
        self.outlier_report = {"enabled": False, "method": "none", "removed_rows": 0}
        has_treatment = treatment_enabled(self.noise_config, self.outlier_config)
        noise_treater: Optional[NoiseTreater] = None
        outlier_treater: Optional[OutlierTreater] = None

        if has_treatment:
            noise_treater = NoiseTreater(self.noise_config).fit(baseline_train)
            treated_train = noise_treater.transform(baseline_train)
            treated_test = noise_treater.transform(baseline_test)
            self.noise_report = noise_treater.describe()

            outlier_treater = OutlierTreater(
                self.outlier_config, random_state=self.random_state
            ).fit(treated_train)
            treated_train, removed = outlier_treater.transform(treated_train)
            treated_test, _ = outlier_treater.transform(treated_test)

            if removed.any() and not self._removal_is_safe(
                split.y_train, removed, split.task_type
            ):
                # Never drop a class below two training rows: fall back to the
                # un-removed frame and record why.
                outlier_fallback = dict(outlier_treater.describe())
                outlier_fallback["removal_skipped"] = True
                outlier_fallback["removed_rows"] = 0
                self.outlier_report = outlier_fallback
                treated_train = noise_treater.transform(baseline_train)
                treated_y_train = split.y_train
                removed_total = 0
            else:
                self.outlier_report = outlier_treater.describe()
                removed_total = int(removed.sum())
                treated_y_train = split.y_train.loc[~removed]

        # 3. Decide which variants this run trains.
        build_baseline = (not has_treatment) or (self.compare_noise and has_treatment)
        build_treated = has_treatment
        variant_frames: Dict[str, tuple] = {}
        if build_baseline:
            variant_frames[VARIANT_BASELINE] = (
                baseline_train,
                baseline_test,
                split.y_train,
                split.y_test,
            )
        if build_treated:
            variant_frames[VARIANT_TREATED] = (
                treated_train,
                treated_test,
                treated_y_train,
                split.y_test,
            )

        # 4. Preprocess each variant independently.
        self.variants = {}
        for name, (X_tr, X_te, y_tr, y_te) in variant_frames.items():
            preprocessor = DataPreprocessor(
                scale_numeric=self.scale_numeric,
                numeric_imputer=self.numeric_imputer,
                categorical_imputer=self.categorical_imputer,
                imputer_params=self.imputer_params,
            )
            X_tr_proc, y_tr_proc = preprocessor.fit_transform(X_tr, y_tr, task=self.task_type)
            X_te_proc, y_te_proc = preprocessor.transform(X_te, y_te, task=self.task_type)
            self.variants[name] = VariantData(
                name=name,
                preprocessor=preprocessor,
                X_train=X_tr_proc,
                X_test=X_te_proc,
                y_train=y_tr_proc,
                y_test=y_te_proc,
                feature_names=preprocessor.feature_names_out,
                n_train=len(X_tr_proc),
                n_test=len(X_te_proc),
                rows_removed=removed_total if name == VARIANT_TREATED else 0,
                noise_treater=noise_treater if name == VARIANT_TREATED else None,
                outlier_treater=outlier_treater if name == VARIANT_TREATED else None,
            )

        first = next(iter(self.variants.values()))
        self.preprocessor = first.preprocessor
        self.X_train_proc = first.X_train
        self.X_test_proc = first.X_test
        self.y_train_proc = first.y_train
        self.y_test_proc = first.y_test
        self.feature_names = first.feature_names

        # 5. Data-quality profile (source + per-variant, before/after treatment).
        self.data_quality = self._build_data_quality(
            baseline_train=baseline_train,
            baseline_test=baseline_test,
            treated_train=treated_train if build_treated else None,
        )

    @staticmethod
    def _removal_is_safe(
        y_train: pd.Series, removed: np.ndarray, task_type: str
    ) -> bool:
        if task_type != TASK_CLASSIFICATION:
            return True
        kept = y_train.loc[~removed]
        if kept.empty:
            return False
        return bool((kept.value_counts() >= 2).all())

    def _build_data_quality(
        self,
        baseline_train: pd.DataFrame,
        baseline_test: pd.DataFrame,
        treated_train: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        source = self.split.clean_frame if self.split and self.split.clean_frame is not None else None
        if source is None:
            source = self._reconstruct_frame()
        quality = profile_dataset(source, self.target_col, self.task_type)
        quality["columns"] = profile_columns(source)
        quality["injection"] = self.injection_report
        quality["treatment"] = {
            "noise": self.noise_report,
            "outliers": self.outlier_report,
        }
        quality["split"] = {
            "id": self.split.split_id if self.split else None,
            "source": self.split.source if self.split else "created",
            "fingerprint": self.split.fingerprint[:16] if self.split else "",
            "test_size": self.test_size,
            "random_state": self.random_state,
            "max_samples": self.max_samples,
            "sampling_strategy": self.sampling_strategy,
        }

        variant_profiles: Dict[str, Any] = {
            VARIANT_BASELINE: {
                "n_train": len(baseline_train),
                "n_test": len(baseline_test),
                "columns": profile_columns(baseline_train),
            }
        }
        if treated_train is not None:
            variant_profiles[VARIANT_TREATED] = {
                "n_train": len(treated_train),
                "n_test": len(baseline_test),
                "columns": profile_columns(treated_train),
                "rows_removed": int(self.outlier_report.get("removed_rows", 0)),
            }
        quality["variants"] = variant_profiles

        if treated_train is not None:
            treated_columns = numeric_columns(
                baseline_train,
                self.noise_config.columns or self.outlier_config.columns,
            )
            attach_histograms(quality, baseline_train, treated_train, treated_columns)
        return quality

    def _reconstruct_frame(self) -> pd.DataFrame:
        """Rebuild the cleaned source frame from the split partitions."""
        assert self.X_train_raw is not None and self.y_train_raw is not None
        train = self.X_train_raw.copy()
        train[self.target_col] = self.y_train_raw
        test = self.X_test_raw.copy()
        test[self.target_col] = self.y_test_raw
        combined = pd.concat([train, test])
        if combined.index.is_unique:
            combined = combined.sort_index()
        return combined.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Trainers
    # ------------------------------------------------------------------
    def get_trainer(self, model_id: str, variant: Optional[str] = None) -> Any:
        """Fetch a fitted trainer, defaulting to the first available variant."""
        name = variant or (self.variant_names[0] if self.variant_names else None)
        if name is None or name not in self.fitted_trainers:
            raise KeyError(f"No fitted trainers for variant {name!r}.")
        trainers = self.fitted_trainers[name]
        if model_id not in trainers:
            raise KeyError(f"Model {model_id!r} was not trained for variant {name!r}.")
        return trainers[model_id]

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
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
        if not self.variants:
            self.prepare()

        selected_ids = [mid for mid in (model_ids or list(MODEL_REGISTRY.keys())) if mid in MODEL_REGISTRY]
        custom_params = custom_params or {}

        if self.numeric_imputer == "none":
            unsupported = [mid for mid in selected_ids if mid not in NAN_CAPABLE_MODELS]
            if unsupported:
                names = ", ".join(MODEL_REGISTRY[mid].display_name for mid in unsupported)
                raise ValueError(
                    "Numeric imputation is set to 'none', but these models cannot "
                    f"consume NaN values: {names}. Pick an imputation strategy or "
                    "select only XGBoost / LightGBM / CatBoost."
                )

        # 1. Initialize Experiment in DB
        exp_name = run_name or f"{dataset_name}_{self.task_type}_{int(time.time())}"
        first_variant = next(iter(self.variants.values()))
        experiment_id = save_experiment(
            run_name=exp_name,
            dataset_name=dataset_name,
            task_type=self.task_type,
            target_col=self.target_col,
            n_train=first_variant.n_train,
            n_test=first_variant.n_test,
            n_features=len(first_variant.feature_names),
            feature_names=first_variant.feature_names,
            split_id=self.split.split_id if self.split else self.split_id,
            comparison_mode=self.comparison_mode,
            noise_config={
                "noise": self.noise_report,
                "outliers": self.outlier_report,
                "injection": self.injection_report,
                "compare_noise": self.comparison_mode,
            },
            imputer_config={
                "numeric": self.numeric_imputer,
                "categorical": self.categorical_imputer,
                "params": self.imputer_params,
                "scale_numeric": self.scale_numeric,
            },
            data_quality=self.data_quality,
        )

        results_by_variant: Dict[str, List[Dict[str, Any]]] = {}
        total_steps = max(1, len(self.variants) * len(selected_ids))
        step = 0

        for variant_name, variant in self.variants.items():
            self.fitted_trainers.setdefault(variant_name, {})
            results_by_variant[variant_name] = []

            for mid in selected_ids:
                pct = step / total_steps
                if progress_cb:
                    label = MODEL_REGISTRY[mid].display_name
                    if self.comparison_mode:
                        label = f"{label} · {variant_name}"
                    progress_cb(mid, pct, f"Training {label}...")
                step += 1

                try:
                    result = self._train_one(
                        experiment_id=experiment_id,
                        variant=variant,
                        model_id=mid,
                        params=custom_params.get(mid),
                    )
                    if result:
                        self.fitted_trainers[variant_name][mid] = self._last_trainer
                        results_by_variant[variant_name].append(result)
                except Exception as exc:  # keep training the remaining models
                    print(f"Error training {mid} ({variant_name}): {exc}")

        if progress_cb:
            progress_cb("done", 1.0, "Training completed!")

        flat_results = [
            result
            for variant_results in results_by_variant.values()
            for result in variant_results
        ]
        comparison = None
        if self.comparison_mode and VARIANT_TREATED in results_by_variant:
            baseline_rows = [
                self._result_to_row(result) for result in results_by_variant[VARIANT_BASELINE]
            ]
            treated_rows = [
                self._result_to_row(result) for result in results_by_variant[VARIANT_TREATED]
            ]
            comparison = compare_variants(
                baseline_rows,
                treated_rows,
                primary_metric_for(self.task_type),
                higher_is_better=True,
            )

        return {
            "experiment_id": experiment_id,
            "run_name": exp_name,
            "task_type": self.task_type,
            "results": flat_results,
            "results_by_variant": results_by_variant,
            "variants": list(self.variants.keys()),
            "comparison": comparison,
            "comparison_mode": self.comparison_mode,
            "n_train": first_variant.n_train,
            "n_test": first_variant.n_test,
            "split_id": self.split.split_id if self.split else None,
            "split_source": self.split.source if self.split else "created",
            "data_quality": self.data_quality,
        }

    # ------------------------------------------------------------------
    # Single model training
    # ------------------------------------------------------------------
    def _train_one(
        self,
        experiment_id: int,
        variant: VariantData,
        model_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        trainer = instantiate_trainer(model_id=model_id, task=self.task_type, params=params)
        t0 = time.perf_counter()
        trainer.fit(variant.X_train, variant.y_train)
        elapsed = round(time.perf_counter() - t0, 4)
        self._last_trainer = trainer

        y_pred = trainer.predict(variant.X_test)
        y_prob = (
            trainer.predict_proba(variant.X_test)
            if self.task_type == TASK_CLASSIFICATION
            else None
        )

        class_names = (
            [str(c) for c in variant.preprocessor.classes_]
            if variant.preprocessor.classes_ is not None
            else None
        )
        metrics = evaluate_model(
            task=self.task_type,
            y_true=variant.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=class_names,
        )

        self._add_overfitting_diagnostics(metrics, trainer, variant)

        importances = trainer.get_feature_importances(variant.feature_names)

        artifact_path = ARTIFACTS_DIR / f"exp_{experiment_id}_{variant.name}_{model_id}.joblib"
        try:
            import joblib

            joblib.dump(
                {"model": trainer.model, "preprocessor": variant.preprocessor},
                artifact_path,
            )
            metrics["artifact_path"] = str(artifact_path)
        except Exception:
            pass

        predictions_sample = self._prediction_sample(variant, y_pred)

        res_id = save_model_result(
            experiment_id=experiment_id,
            model_id=model_id,
            model_name=trainer.display_name,
            category=trainer.category,
            train_time_sec=elapsed,
            metrics=metrics,
            params=trainer.params,
            feature_importances=importances,
            predictions_sample=predictions_sample,
            variant=variant.name,
        )

        return {
            "result_id": res_id,
            "model_id": model_id,
            "model_name": trainer.display_name,
            "category": trainer.category,
            "variant": variant.name,
            "train_time_sec": elapsed,
            "metrics": metrics,
            "feature_importances": importances,
        }

    def _add_overfitting_diagnostics(
        self, metrics: Dict[str, Any], trainer: Any, variant: VariantData
    ) -> None:
        try:
            y_train_pred = trainer.predict(variant.X_train)
            if self.task_type == TASK_CLASSIFICATION:
                from sklearn.metrics import accuracy_score, f1_score

                tr_f1 = round(
                    float(
                        f1_score(
                            variant.y_train, y_train_pred, average="macro", zero_division=0
                        )
                    ),
                    4,
                )
                tr_acc = round(float(accuracy_score(variant.y_train, y_train_pred)), 4)
                metrics["train_f1_macro"] = tr_f1
                metrics["train_accuracy"] = tr_acc
                gap = round(tr_f1 - float(metrics.get("f1_macro") or 0.0), 4)
            else:
                from sklearn.metrics import mean_squared_error, r2_score

                tr_r2 = round(float(r2_score(variant.y_train, y_train_pred)), 4)
                tr_rmse = round(
                    float(np.sqrt(mean_squared_error(variant.y_train, y_train_pred))), 4
                )
                metrics["train_r2"] = tr_r2
                metrics["train_rmse"] = tr_rmse
                gap = round(tr_r2 - float(metrics.get("r2") or 0.0), 4)
            metrics["overfit_gap"] = gap
            metrics["is_overfitting"] = bool(gap >= 0.15)
        except Exception:
            metrics["is_overfitting"] = False

    def _prediction_sample(
        self, variant: VariantData, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        n_sample = min(50, len(variant.y_test))
        actual_sample = variant.y_test[:n_sample].tolist()
        pred_sample = y_pred[:n_sample].tolist()

        if self.task_type == TASK_CLASSIFICATION and variant.preprocessor.label_encoder:
            try:
                actual_sample = variant.preprocessor.inverse_transform_target(
                    variant.y_test[:n_sample]
                ).tolist()
                pred_sample = variant.preprocessor.inverse_transform_target(
                    y_pred[:n_sample]
                ).tolist()
            except Exception:
                pass

        return {"actual": actual_sample, "predicted": pred_sample}

    @staticmethod
    def _result_to_row(result: Dict[str, Any]) -> Dict[str, Any]:
        """Shape a result dict like a stored leaderboard row for comparisons."""
        return {
            "model_id": result.get("model_id"),
            "model_name": result.get("model_name"),
            "category": result.get("category"),
            "train_time_sec": result.get("train_time_sec"),
            "metrics": result.get("metrics") or {},
        }


# ------------------------------------------------------------------------------
# Config coercion helpers
# ------------------------------------------------------------------------------
def _coerce_noise_config(value: Any) -> NoiseConfig:
    if isinstance(value, NoiseConfig):
        return value
    return NoiseConfig.from_dict(value)


def _coerce_outlier_config(value: Any) -> OutlierConfig:
    if isinstance(value, OutlierConfig):
        return value
    return OutlierConfig.from_dict(value)


def _coerce_injection_config(value: Any) -> InjectionConfig:
    if isinstance(value, InjectionConfig):
        return value
    return InjectionConfig.from_dict(value)
