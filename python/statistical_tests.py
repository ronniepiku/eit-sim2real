"""Statistical testing for EIT touch classification experiments.

Implements 5-fold stratified cross-validation with paired t-tests and
Bonferroni correction, as described in the methodology chapter.

Reports:
- Mean +/- std accuracy/F1 per condition
- Paired t-test p-values between conditions
- Cohen's d effect sizes
- Bonferroni-corrected significance

Usage:
    uv run python/statistical_tests.py --data-path data/eit_dataset.mat
    uv run python/statistical_tests.py --model cnn1d --data-path data/eit_dataset.mat
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from configs.loader import load_config
from data.load_dataset import get_cv_splits, load_mat_dataset
from models.baselines import get_baseline, train_baseline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for paired samples."""
    diff = x - y
    return (
        float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    )


def paired_t_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Perform paired t-test, returning (t_statistic, p_value)."""
    from scipy import stats

    t_stat, p_val = stats.ttest_rel(x, y)
    return float(t_stat), float(p_val)


def run_cv_experiment(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    n_folds: int = 5,
    seed: int = 42,
    scaler_type: str = "robust",
) -> dict[str, np.ndarray]:
    """Run k-fold CV on training data, evaluate on held-out test set.

    Returns per-fold accuracy and F1 on the test set.
    """
    folds = get_cv_splits(
        X_train_full, y_train_full, n_folds=n_folds, random_state=seed
    )

    accuracies = np.zeros(n_folds)
    f1_scores = np.zeros(n_folds)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr = X_train_full[train_idx]
        y_tr = y_train_full[train_idx]

        # Normalise per fold
        scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_test)

        if model_name == "cnn1d":
            import torch
            from train import train_cnn

            cfg = load_config()
            X_val_fold = X_train_full[val_idx]
            y_val_fold = y_train_full[val_idx]
            X_val_fold = scaler.transform(X_val_fold)

            torch.manual_seed(seed + fold_idx)
            model, _ = train_cnn(
                X_tr,
                y_tr,
                X_val_fold,
                y_val_fold,
                epochs=cfg["training"]["epochs"],
                batch_size=cfg["training"]["batch_size"],
                lr=cfg["training"]["learning_rate"],
                weight_decay=cfg["training"]["weight_decay"],
                scheduler_patience=cfg["training"]["scheduler_patience"],
                scheduler_factor=cfg["training"]["scheduler_factor"],
                early_stopping_patience=cfg["training"]["early_stopping_patience"],
            )
            model.eval()
            X_tensor = torch.from_numpy(X_te).float()
            with torch.no_grad():
                y_pred = model(X_tensor).argmax(dim=1).cpu().numpy()
        else:
            model = get_baseline(model_name, random_state=seed + fold_idx)
            model = train_baseline(model, X_tr, y_tr)
            y_pred = model.predict(X_te)

        accuracies[fold_idx] = accuracy_score(y_test, y_pred)
        f1_scores[fold_idx] = f1_score(y_test, y_pred, average="macro")

        logger.info(
            f"  Fold {fold_idx + 1}/{n_folds}: "
            f"acc={accuracies[fold_idx]:.4f}, f1={f1_scores[fold_idx]:.4f}"
        )

    return {"accuracies": accuracies, "f1_scores": f1_scores}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Statistical testing with 5-fold CV.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["svm", "random_forest", "mlp", "cnn1d"],
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/statistics"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    data_path = args.data_path or Path(cfg["data"]["path"])
    scaler_type = cfg.get("data", {}).get("scaler", "robust")
    n_folds = args.n_folds

    np.random.seed(args.seed)

    # Load both clean and noisy data
    X_clean, y = load_mat_dataset(data_path, use_noisy=False)
    X_noisy, _ = load_mat_dataset(data_path, use_noisy=True)

    # Use a fixed test set (15%) and run CV on the rest
    from sklearn.model_selection import train_test_split

    test_size = cfg["data"].get("test_size", 0.15)

    X_clean_trainval, X_clean_test, y_trainval, y_test = train_test_split(
        X_clean, y, test_size=test_size, random_state=args.seed, stratify=y
    )
    X_noisy_trainval, X_noisy_test, _, _ = train_test_split(
        X_noisy, y, test_size=test_size, random_state=args.seed, stratify=y
    )

    # Run 4 conditions
    conditions = {
        "clean_train_clean_test": (X_clean_trainval, y_trainval, X_clean_test, y_test),
        "clean_train_noisy_test": (X_clean_trainval, y_trainval, X_noisy_test, y_test),
        "noisy_train_noisy_test": (X_noisy_trainval, y_trainval, X_noisy_test, y_test),
        "noisy_train_clean_test": (X_noisy_trainval, y_trainval, X_clean_test, y_test),
    }

    all_results: dict[str, dict] = {}

    for cond_name, (X_tr, y_tr, X_te, y_te) in conditions.items():
        logger.info(f"\nCondition: {cond_name}")
        results = run_cv_experiment(
            X_tr,
            y_tr,
            X_te,
            y_te,
            model_name=args.model,
            n_folds=n_folds,
            seed=args.seed,
            scaler_type=scaler_type,
        )
        all_results[cond_name] = {
            "accuracy_mean": float(np.mean(results["accuracies"])),
            "accuracy_std": float(np.std(results["accuracies"])),
            "f1_mean": float(np.mean(results["f1_scores"])),
            "f1_std": float(np.std(results["f1_scores"])),
            "per_fold_accuracy": results["accuracies"].tolist(),
            "per_fold_f1": results["f1_scores"].tolist(),
        }
        logger.info(
            f"  Mean acc: {all_results[cond_name]['accuracy_mean']:.4f} "
            f"± {all_results[cond_name]['accuracy_std']:.4f}"
        )

    # Paired t-tests with Bonferroni correction
    comparisons = [
        ("clean_train_noisy_test", "noisy_train_noisy_test"),  # Key comparison
        ("clean_train_clean_test", "noisy_train_clean_test"),
        ("clean_train_clean_test", "clean_train_noisy_test"),  # Vulnerability gap
    ]
    n_comparisons = len(comparisons)
    alpha = 0.05

    statistical_tests = []
    for cond_a, cond_b in comparisons:
        acc_a = np.array(all_results[cond_a]["per_fold_accuracy"])
        acc_b = np.array(all_results[cond_b]["per_fold_accuracy"])
        t_stat, p_val = paired_t_test(acc_a, acc_b)
        d = cohens_d(acc_a, acc_b)
        p_corrected = min(p_val * n_comparisons, 1.0)  # Bonferroni
        significant = p_corrected < alpha

        test_result = {
            "comparison": f"{cond_a} vs {cond_b}",
            "t_statistic": t_stat,
            "p_value": p_val,
            "p_corrected_bonferroni": p_corrected,
            "cohens_d": d,
            "significant": significant,
            "alpha": alpha,
            "n_comparisons": n_comparisons,
        }
        statistical_tests.append(test_result)
        logger.info(
            f"\n  {cond_a} vs {cond_b}:\n"
            f"    t={t_stat:.4f}, p={p_val:.6f}, "
            f"p_corrected={p_corrected:.6f}, d={d:.4f}, "
            f"significant={significant}"
        )

    # Save all results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model,
        "n_folds": n_folds,
        "seed": args.seed,
        "conditions": all_results,
        "statistical_tests": statistical_tests,
    }
    output_path = args.output_dir / f"{args.model}_statistical_tests.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"Statistical Testing Summary: {args.model}")
    print(f"{'=' * 60}")
    print(f"{'Condition':<30} {'Accuracy':>12} {'F1 (macro)':>12}")
    print(f"{'-' * 60}")
    for cond, res in all_results.items():
        print(
            f"{cond:<30} "
            f"{res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f} "
            f"{res['f1_mean']:.4f}±{res['f1_std']:.4f}"
        )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
