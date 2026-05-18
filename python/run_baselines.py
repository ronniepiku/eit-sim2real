"""Run all baseline experiments for Week 5.

Trains SVM, Random Forest, and MLP under 4 conditions:
1. Clean → Clean  (ceiling)
2. Clean → Noisy  (vulnerability)
3. Noisy → Noisy  (robustness)
4. Noisy → Clean  (generalisation)

Records accuracy and macro-F1 for each combination and writes a
summary CSV to ``results/tables/baseline_results.csv``.

Usage:
    uv run python/run_baselines.py
    uv run python/run_baselines.py --data-path data/eit_dataset.mat
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from configs.loader import load_config
from data.load_dataset import load_mat_dataset, prepare_splits
from models.baselines import get_baseline, train_baseline
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS = ["svm", "random_forest", "mlp"]

CONDITIONS = [
    ("clean_train_clean_eval", "clean", "clean"),
    ("clean_train_noisy_eval", "clean", "noisy"),
    ("noisy_train_noisy_eval", "noisy", "noisy"),
    ("noisy_train_clean_eval", "noisy", "clean"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all baseline experiments.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--results-csv", type=Path, default=Path("results/tables/baseline_results.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    seed = args.seed
    data_path = args.data_path or Path(cfg["data"]["path"])
    np.random.seed(seed)

    # Load both clean and noisy
    logger.info(f"Loading dataset from {data_path}")
    X_noisy, y = load_mat_dataset(data_path, use_noisy=True)
    X_clean, _ = load_mat_dataset(data_path, use_noisy=False)

    # Prepare splits (same seed ensures identical indices)
    dataset_clean = prepare_splits(X_clean, y, random_state=seed)
    dataset_noisy = prepare_splits(X_noisy, y, random_state=seed)

    splits = {
        "clean": dataset_clean,
        "noisy": dataset_noisy,
    }

    results: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        for condition_name, train_key, eval_key in CONDITIONS:
            logger.info(f"── {model_name} | {condition_name} ──")

            ds_train = splits[train_key]
            ds_eval = splits[eval_key]

            model = get_baseline(model_name, random_state=seed)
            model = train_baseline(model, ds_train.X_train, ds_train.y_train)

            # Evaluate on test split
            y_pred = model.predict(ds_eval.X_test)
            acc = accuracy_score(ds_eval.y_test, y_pred)
            f1 = f1_score(ds_eval.y_test, y_pred, average="macro")

            logger.info(f"  Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

            results.append({
                "model": model_name,
                "condition": condition_name,
                "train_data": train_key,
                "eval_data": eval_key,
                "accuracy": acc,
                "f1_macro": f1,
            })

            # Save model
            model_path = args.output_dir / f"{model_name}_{train_key}.joblib"
            if not model_path.exists():
                joblib.dump(model, model_path)

    # Save results
    df = pd.DataFrame(results)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results_csv, index=False)
    logger.info(f"\nResults saved to {args.results_csv}")

    # Print summary table
    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 70)
    pivot = df.pivot_table(
        index="model", columns="condition", values=["accuracy", "f1_macro"]
    )
    print(pivot.round(4).to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
