"""
Smoke test script for ML2 Pipeline & SQLite persistence.
Tests all 8 models on both Classification and Regression datasets.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.datasets import make_classification, make_regression

from core.pipeline import TrainingPipeline
from core.storage import get_all_experiments, get_leaderboard
from models.registry import MODEL_REGISTRY


def test_classification():
    print("\n--- Running Classification Pipeline Test ---")
    X, y = make_classification(
        n_samples=150, n_features=6, n_informative=4, n_classes=2, random_state=42
    )
    cols = [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    pipeline = TrainingPipeline(
        df=df,
        target_col="target",
        task="classification",
        test_size=0.25,
        random_state=42,
    )
    res = pipeline.run_benchmark(
        model_ids=list(MODEL_REGISTRY.keys()),
        run_name="Test_Classification_Benchmark",
        dataset_name="Synthetic_Classification",
    )

    exp_id = res["experiment_id"]
    print(f"Classification Experiment ID: {exp_id}")
    leaderboard = get_leaderboard(exp_id)
    print(f"Successfully trained {len(leaderboard)} models:")
    for m in leaderboard:
        acc = m["metrics"].get("accuracy")
        f1 = m["metrics"].get("f1_macro")
        t = m["train_time_sec"]
        print(f"  - {m['model_name']:25} | Acc: {acc:.4f} | F1: {f1:.4f} | Time: {t:.4f}s")
    assert len(leaderboard) == len(MODEL_REGISTRY), "Not all models completed classification!"


def test_regression():
    print("\n--- Running Regression Pipeline Test ---")
    X, y = make_regression(
        n_samples=150, n_features=6, n_informative=4, noise=0.1, random_state=42
    )
    cols = [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["price"] = y

    pipeline = TrainingPipeline(
        df=df,
        target_col="price",
        task="regression",
        test_size=0.25,
        random_state=42,
    )
    res = pipeline.run_benchmark(
        model_ids=list(MODEL_REGISTRY.keys()),
        run_name="Test_Regression_Benchmark",
        dataset_name="Synthetic_Regression",
    )

    exp_id = res["experiment_id"]
    print(f"Regression Experiment ID: {exp_id}")
    leaderboard = get_leaderboard(exp_id)
    print(f"Successfully trained {len(leaderboard)} models:")
    for m in leaderboard:
        r2 = m["metrics"].get("r2")
        rmse = m["metrics"].get("rmse")
        t = m["train_time_sec"]
        print(f"  - {m['model_name']:25} | R2: {r2:.4f} | RMSE: {rmse:.4f} | Time: {t:.4f}s")
    assert len(leaderboard) == len(MODEL_REGISTRY), "Not all models completed regression!"


if __name__ == "__main__":
    try:
        test_classification()
        test_regression()
        print("\nAll pipeline tests PASSED successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
