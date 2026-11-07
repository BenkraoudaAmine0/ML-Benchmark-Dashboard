"""
Verification script using real repository datasets:
1. Heart Attack dataset for Classification.
2. Weather dataset for Regression.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from Modeling.core.data_loader import load_dataset
from Modeling.core.pipeline import TrainingPipeline
from Modeling.core.storage import get_all_experiments, get_leaderboard
from Modeling.models.registry import MODEL_REGISTRY


def run_heart_classification():
    heart_path = PROJECT_ROOT / "Datasets" / "Heart_Attack_Analysis" / "heart.csv"
    if not heart_path.exists():
        print(f"Skipping heart: {heart_path} not found")
        return

    print("\n=======================================================")
    print("Benchmark 1: Heart Attack Analysis (Classification)")
    print("=======================================================")
    df = load_dataset(heart_path)
    print(f"Dataset shape: {df.shape}")

    pipeline = TrainingPipeline(
        df=df,
        target_col="output",
        task="classification",
        test_size=0.2,
        random_state=42,
    )
    res = pipeline.run_benchmark(
        model_ids=list(MODEL_REGISTRY.keys()),
        run_name="Heart_Attack_Classification_Benchmark",
        dataset_name="Heart_Attack_Analysis",
    )
    exp_id = res["experiment_id"]
    leaderboard = get_leaderboard(exp_id)
    print(f"\nLeaderboard for Experiment #{exp_id}:")
    for m in leaderboard:
        print(
            f"  {m['model_name']:25} | "
            f"Acc: {m['metrics'].get('accuracy', 0):.4f} | "
            f"F1: {m['metrics'].get('f1_macro', 0):.4f} | "
            f"ROC-AUC: {m['metrics'].get('roc_auc') or 0:.4f} | "
            f"Time: {m['train_time_sec']:.4f}s"
        )


def run_weather_regression():
    weather_path = PROJECT_ROOT / "Datasets" / "weather-dataset" / "weatherHistory.csv"
    if not weather_path.exists():
        print(f"Skipping weather: {weather_path} not found")
        return

    print("\n=======================================================")
    print("Benchmark 2: Weather History (Regression)")
    print("=======================================================")
    # Sample down to 2000 rows for quick benchmark verification
    df = load_dataset(weather_path).sample(n=2000, random_state=42).reset_index(drop=True)
    drop_cols = ["Formatted Date", "Daily Summary"]
    print(f"Dataset shape (sampled): {df.shape}")

    pipeline = TrainingPipeline(
        df=df,
        target_col="Temperature (C)",
        task="regression",
        test_size=0.2,
        random_state=42,
        drop_cols=drop_cols,
    )
    res = pipeline.run_benchmark(
        model_ids=list(MODEL_REGISTRY.keys()),
        run_name="Weather_Temperature_Regression_Benchmark",
        dataset_name="Weather_History",
    )
    exp_id = res["experiment_id"]
    leaderboard = get_leaderboard(exp_id)
    print(f"\nLeaderboard for Experiment #{exp_id}:")
    for m in leaderboard:
        print(
            f"  {m['model_name']:25} | "
            f"R2: {m['metrics'].get('r2', 0):.4f} | "
            f"RMSE: {m['metrics'].get('rmse', 0):.4f} | "
            f"MAE: {m['metrics'].get('mae', 0):.4f} | "
            f"Time: {m['train_time_sec']:.4f}s"
        )


if __name__ == "__main__":
    run_heart_classification()
    run_weather_regression()
    print("\nDatabase verification:")
    all_exps = get_all_experiments()
    print(f"Total experiments recorded in SQLite: {len(all_exps)}")

