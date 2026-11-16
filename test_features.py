"""
Smoke tests for imputation, noise treatment, persistent splits, variant
comparisons and the AI context builder.

No network calls are made: the AI report generator is exercised only through its
pure helpers (normalisation and Markdown rendering).

Run with:  python test_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.ai_insights import normalize_report, report_to_markdown, resolve_settings
from core.data_quality import compare_variants, profile_series
from core.noise import (
    InjectionConfig,
    NoiseConfig,
    NoiseTreater,
    OutlierConfig,
    OutlierTreater,
    inject_noise,
)
from core.pipeline import TrainingPipeline
from core.preprocessor import DataPreprocessor
from core.splits import split_or_create
from core.storage import (
    delete_experiment,
    delete_split,
    get_experiment,
    get_leaderboard,
    get_variants,
    list_splits,
)
from frontend.insight_context import build_insight_context

IMPUTERS = [
    "median",
    "mean",
    "most_frequent",
    "constant",
    "interpolate",
    "ffill",
    "knn",
    "iterative",
    "iterative_rf",
]


def _frame(n: int = 140, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)]
    )
    df["cat"] = rng.choice(["a", "b", "c"], size=n)
    df.loc[::7, "f0"] = np.nan
    df.loc[::11, "f1"] = np.nan
    df.loc[::13, "cat"] = np.nan
    df["target"] = rng.integers(0, 2, size=n)
    return df


def test_imputers() -> None:
    print("\n--- Imputation strategies ---")
    df = _frame()
    X_train, X_test = df.drop(columns=["target"]).iloc[:100], df.drop(columns=["target"]).iloc[100:]
    y_train = df["target"].iloc[:100]

    for strategy in IMPUTERS:
        for categorical in ("most_frequent", "constant"):
            pre = DataPreprocessor(
                scale_numeric=True,
                numeric_imputer=strategy,
                categorical_imputer=categorical,
                imputer_params={"add_indicator": strategy == "constant"},
            )
            X_tr, _ = pre.fit_transform(X_train, y_train, task="classification")
            X_te, _ = pre.transform(X_test, task="classification")
            assert not np.isnan(X_tr).any(), f"{strategy}: train still has NaN"
            assert not np.isnan(X_te).any(), f"{strategy}: test still has NaN"
            assert X_tr.shape[1] == X_te.shape[1], f"{strategy}: shape mismatch"
            assert len(pre.feature_names_out) == X_tr.shape[1], (
                f"{strategy}/{categorical}: feature-name mismatch "
                f"({len(pre.feature_names_out)} names for {X_tr.shape[1]} columns)"
            )
        print(f"  OK - {strategy}")

    # 'none' is a pass-through and is allowed to keep NaN for NaN-native models.
    pre = DataPreprocessor(numeric_imputer="none")
    X_tr, _ = pre.fit_transform(X_train, y_train, task="classification")
    assert np.isnan(X_tr).any(), "none should preserve NaN"
    print("  OK -none (NaN preserved)")


def test_noise_treatment() -> None:
    print("\n--- Noise treatment & outliers ---")
    rng = np.random.default_rng(3)
    n = 300
    signal = np.sin(np.linspace(0, 12 * np.pi, n))
    noisy = signal + rng.normal(0, 0.35, size=n)
    spikes = noisy.copy()
    spike_idx = rng.choice(n, size=12, replace=False)
    spikes[spike_idx] += 8.0
    df = pd.DataFrame({"signal": noisy, "spiky": spikes, "other": rng.normal(size=n)})

    before = profile_series(df["signal"])["hf_noise_ratio"]
    treater = NoiseTreater(NoiseConfig(strategy="rolling_mean", window=7)).fit(df)
    smoothed = treater.transform(df)
    after = profile_series(smoothed["signal"])["hf_noise_ratio"]
    assert after < before, f"rolling mean did not reduce noise ({before} -> {after})"
    print(f"  OK -rolling mean lowers HF noise ratio ({before:.3f} -> {after:.3f})")

    hampel = NoiseTreater(NoiseConfig(strategy="hampel", window=7, k=3.0)).fit(df)
    cleaned = hampel.transform(df)
    residual_before = np.abs(df["spiky"].to_numpy() - signal).max()
    residual_after = np.abs(cleaned["spiky"].to_numpy() - signal).max()
    assert residual_after <= residual_before, "Hampel should not increase spikes"
    print(f"  OK -Hampel caps spikes (max dev {residual_before:.2f} -> {residual_after:.2f})")

    clipped = NoiseTreater(NoiseConfig(strategy="iqr_clip", k=1.5)).fit(df)
    clipped_df = clipped.transform(df)
    q1, q3 = df["signal"].quantile(0.25), df["signal"].quantile(0.75)
    iqr = q3 - q1
    assert clipped_df["signal"].max() <= q3 + 1.5 * iqr + 1e-9
    assert clipped_df["signal"].min() >= q1 - 1.5 * iqr - 1e-9
    print("  OK -IQR clipping respects the fences")

    outlier = OutlierTreater(OutlierConfig(method="iqr", action="remove", k=1.5)).fit(df)
    treated, removed = outlier.transform(df)
    assert removed.sum() > 0, "expected flagged outlier rows"
    assert len(treated) == len(df) and len(treated) - removed.sum() == (~removed).sum()
    print(f"  OK -outlier removal flags {int(removed.sum())} rows (test rows untouched)")

    cfg = InjectionConfig(method="gaussian", ratio=0.1, seed=11)
    out_a, report_a = inject_noise(df, cfg)
    out_b, report_b = inject_noise(df, cfg)
    pd.testing.assert_frame_equal(out_a, out_b)
    assert report_a["cells_changed"] == report_b["cells_changed"] > 0
    print("  OK -injection is deterministic for a fixed seed")

    missing_cfg = InjectionConfig(method="missing", ratio=0.1, seed=5)
    out_missing, report_m = inject_noise(df, missing_cfg)
    assert out_missing.isna().sum().sum() == report_m["cells_changed"]
    print("  OK -missing-value injection matches its report")

    # NaN masks survive smoothing so imputation still sees missing values.
    with_nan = df.copy()
    with_nan.loc[10:20, "signal"] = np.nan
    masked = NoiseTreater(NoiseConfig(strategy="rolling_mean", window=5)).fit(with_nan).transform(with_nan)
    assert masked.loc[10:20, "signal"].isna().all(), "NaN mask must be preserved"
    print("  OK -smoothing preserves the missing-value mask")


def test_splits() -> None:
    print("\n--- Persistent splits ---")
    for leftover in list_splits(dataset_name="test_features_split"):
        delete_split(leftover["id"])
    df = _frame(n=200, seed=21)
    common = dict(
        target_col="target",
        task="classification",
        test_size=0.25,
        random_state=42,
        max_samples=150,
        dataset_name="test_features_split",
        sampling_strategy="random",
    )
    first = split_or_create(df, **common)
    assert first.created and first.split_id is not None
    second = split_or_create(df, **common)
    assert second.split_id == first.split_id, "identical config must reuse the split"
    assert not second.created
    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_series_equal(first.y_test, second.y_test)
    print(f"  OK -identical data + config reuses split #{first.split_id}")

    loaded = split_or_create(df, **common, split_id=first.split_id)
    assert loaded.source == "loaded"
    pd.testing.assert_frame_equal(first.X_train, loaded.X_train)
    print(f"  OK -split #{first.split_id} loads from disk with identical rows")

    changed = split_or_create(df, **{**common, "random_state": 7})
    assert changed.split_id != first.split_id, "a new seed must create a new split"
    print(f"  OK -different seed creates split #{changed.split_id}")

    for split in (first, changed):
        if split.split_id:
            delete_split(split.split_id)
    print("  OK -test splits cleaned up")


def test_comparison_run_and_context() -> None:
    print("\n--- Baseline vs treated run & AI context ---")
    for leftover in list_splits(dataset_name="test_features_comparison"):
        delete_split(leftover["id"])
    df = _frame(n=180, seed=33)
    pipeline = TrainingPipeline(
        df=df,
        target_col="target",
        task="classification",
        test_size=0.25,
        random_state=42,
        dataset_name="test_features_comparison",
        numeric_imputer="knn",
        imputer_params={"n_neighbors": 3},
        noise_config=NoiseConfig(strategy="rolling_mean", window=5),
        outlier_config=OutlierConfig(method="iqr", action="clip", k=1.5),
        injection_config=InjectionConfig(method="gaussian", ratio=0.05, seed=42),
        compare_noise=True,
    )
    results = pipeline.run_benchmark(
        model_ids=["dtree", "knn"],
        run_name="Features_Comparison_Test",
        dataset_name="test_features_comparison",
    )
    exp_id = results["experiment_id"]
    try:
        assert results["comparison_mode"], "comparison mode should be on"
        assert set(results["variants"]) == {"baseline", "treated"}
        assert len(results["results"]) == 4, "2 models x 2 variants"
        assert results["split_id"] is not None

        variants = get_variants(exp_id)
        assert variants == ["baseline", "treated"], variants
        assert len(get_leaderboard(exp_id)) == 4
        assert len(get_leaderboard(exp_id, variant="treated")) == 2
        assert len({row["metrics"]["artifact_path"] for row in get_leaderboard(exp_id) if "artifact_path" in row["metrics"]}) >= 2
        print(f"  OK -experiment #{exp_id}: 4 rows, variants {variants}, split #{results['split_id']}")

        comparison = compare_variants(
            get_leaderboard(exp_id, variant="baseline"),
            get_leaderboard(exp_id, variant="treated"),
            "f1_macro",
        )
        assert comparison["n_models"] == 2
        assert comparison["wins"] + comparison["losses"] + comparison["unchanged"] == 2
        print("  OK -paired comparison produced per-model deltas")

        exp_info = get_experiment(exp_id)
        context = build_insight_context(
            exp_info,
            get_leaderboard(exp_id, variant="baseline"),
            baseline_leaderboard=get_leaderboard(exp_id, variant="baseline"),
            treated_leaderboard=get_leaderboard(exp_id, variant="treated"),
        )
        import json

        payload = json.dumps(context)
        assert len(payload) < 60000, f"AI context too large: {len(payload)} chars"
        for key in ("metric_comparison", "latency", "variant_comparison", "noise_profile"):
            assert key in context["charts"], f"missing chart payload: {key}"
        assert context["comparison"]["n_models"] == 2
        assert context["data_quality"], "data quality should be embedded"
        print(f"  OK -AI context is {len(payload)} chars with {len(context['charts'])} charts")
    finally:
        delete_experiment(exp_id)
        for artifact in (PROJECT_ROOT / "storage" / "artifacts").glob(f"exp_{exp_id}_*.joblib"):
            artifact.unlink(missing_ok=True)
        for leftover in list_splits(dataset_name="test_features_comparison"):
            delete_split(leftover["id"])
        print(f"  OK -experiment #{exp_id} and its split cleaned up")

    # Pure AI helpers (no network).
    report = normalize_report(
        {
            "summary": "S",
            "key_findings": "one",
            "chart_explanations": {"latency": "L"},
            "recommendations": ["R"],
        }
    )
    assert report["key_findings"] == ["one"]
    assert report["chart_explanations"]["latency"] == "L"
    markdown = report_to_markdown(report, {"provider": "groq", "model": "test"})
    assert "## Summary" in markdown and "## Chart explanations" in markdown
    settings = resolve_settings(provider="groq")
    assert settings.base_url.startswith("https://api.groq.com")
    print("  OK -AI helpers normalise and render without network access")


def main() -> None:
    try:
        test_imputers()
        test_noise_treatment()
        test_splits()
        test_comparison_run_and_context()
    except Exception as exc:
        print(f"\nTest failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("\nAll feature tests PASSED.")


if __name__ == "__main__":
    main()
