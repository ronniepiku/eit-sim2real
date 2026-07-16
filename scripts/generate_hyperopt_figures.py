from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from eit_sim2real.configs import load_config
from eit_sim2real.data import load_mat_dataset, prepare_splits
from eit_sim2real.data.noise import NoiseConfig
from eit_sim2real.train import train_cnn
from eit_sim2real.utils import get_device, predict_cnn
from eit_sim2real.visualisation import (
    plot_training_curves,
    plot_confusion_matrix_and_save,
    plot_per_class_metrics_and_save,
)
from eit_sim2real.constants import CLASS_NAMES


def _format_per_class_table(domain_metrics: dict[str, dict[str, float | int]]) -> list[str]:
    lines = [
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name, values in domain_metrics.items():
        lines.append(
            "| "
            f"{class_name} | "
            f"{values['precision']:.4f} | "
            f"{values['recall']:.4f} | "
            f"{values['f1']:.4f} | "
            f"{values['support']} |"
        )
    return lines


def main() -> None:
    root = Path(".")
    fig_dir = root / "results" / "figures" / "hyperparameter_optimisation"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(None)
    report_path = root / "results" / "hyperparameter_optimisation" / "optimisation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    best = report["best_hyperparameters"]

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    # Load raw clean/noisy datasets (hyperopt final protocol)
    X_clean, y = load_mat_dataset(Path("data/eit_dataset.mat"), use_noisy=False)
    X_noisy, _ = load_mat_dataset(Path("data/eit_dataset.mat"), use_noisy=True)

    ds_clean = prepare_splits(X_clean, y, random_state=seed, scaler_type="robust")
    ds_noisy = prepare_splits(X_noisy, y, random_state=seed, scaler_type="robust")

    noise_cfg_section = cfg.get("noise_augmentation", {})
    noise_config = NoiseConfig(
        enabled=True,
        snr_db=noise_cfg_section.get("snr_db", 40.0),
        noise_floor=noise_cfg_section.get("noise_floor", 1e-4),
        contact_impedance_std_percent=noise_cfg_section.get(
            "contact_impedance_std_percent", 10.0
        ),
        max_bias=noise_cfg_section.get("max_bias", 0.02),
        adc_bits=noise_cfg_section.get("adc_bits", 16),
        voltage_range=noise_cfg_section.get("voltage_range", 1.0),
        n_electrodes=noise_cfg_section.get("n_electrodes", 16),
    )
    severity_range_cfg = noise_cfg_section.get("severity_range")
    severity_range = tuple(severity_range_cfg) if severity_range_cfg else (0.5, 2.0)

    device = get_device()
    training_cfg = cfg.get("training", {})
    hp_cfg = cfg.get("hyperparameter_optimisation", {})
    epochs = int(hp_cfg.get("final_model_epochs", training_cfg.get("epochs", 300)))
    scheduler_patience = int(training_cfg.get("scheduler_patience", 10))
    scheduler_factor = float(training_cfg.get("scheduler_factor", 0.5))
    early_stopping_patience = int(
        cfg.get("hyperparameter_optimisation", {}).get(
            "final_early_stopping_patience",
            training_cfg.get("early_stopping_patience", 40),
        )
    )

    # 1) Recreate NOISY optimized model training curve and metrics figures
    model_noisy, hist_noisy = train_cnn(
        ds_noisy.X_train,
        ds_noisy.y_train,
        ds_noisy.X_val,
        ds_noisy.y_val,
        n_classes=len(np.unique(y)),
        epochs=epochs,
        batch_size=int(best["batch_size"]),
        lr=float(best["learning_rate"]),
        weight_decay=float(best["weight_decay"]),
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        early_stopping_patience=early_stopping_patience,
        device=device,
        noise_config=noise_config,
        input_scaler=ds_noisy.scaler,
        severity_range=severity_range,
        dropout=float(best["dropout"]),
        channels=list(best["channels"]),
    )

    plot_training_curves(hist_noisy, fig_dir, "hyperopt_noisy")
    y_pred_noisy = predict_cnn(model_noisy, ds_noisy.X_test, device=device)
    plot_confusion_matrix_and_save(
        ds_noisy.y_test,
        y_pred_noisy,
        fig_dir,
        model_name="cnn1d_hyperopt",
        noise_tag="noisy_domain",
        split_name="test",
    )
    plot_per_class_metrics_and_save(
        ds_noisy.y_test,
        y_pred_noisy,
        fig_dir,
        model_name="cnn1d_hyperopt",
        noise_tag="noisy_domain",
        split_name="test",
    )

    # 2) Recreate CLEAN optimized model training curve and metrics figures
    model_clean, hist_clean = train_cnn(
        ds_clean.X_train,
        ds_clean.y_train,
        ds_clean.X_val,
        ds_clean.y_val,
        n_classes=len(np.unique(y)),
        epochs=epochs,
        batch_size=int(best["batch_size"]),
        lr=float(best["learning_rate"]),
        weight_decay=float(best["weight_decay"]),
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        early_stopping_patience=early_stopping_patience,
        device=device,
        noise_config=None,
        input_scaler=None,
        severity_range=None,
        dropout=float(best["dropout"]),
        channels=list(best["channels"]),
    )

    plot_training_curves(hist_clean, fig_dir, "hyperopt_clean")
    y_pred_clean = predict_cnn(model_clean, ds_clean.X_test, device=device)
    plot_confusion_matrix_and_save(
        ds_clean.y_test,
        y_pred_clean,
        fig_dir,
        model_name="cnn1d_hyperopt",
        noise_tag="clean_domain",
        split_name="test",
    )
    plot_per_class_metrics_and_save(
        ds_clean.y_test,
        y_pred_clean,
        fig_dir,
        model_name="cnn1d_hyperopt",
        noise_tag="clean_domain",
        split_name="test",
    )

    # Save per-class values as JSON for manuscript tables
    prec_n, rec_n, f1_n, sup_n = precision_recall_fscore_support(
        ds_noisy.y_test,
        y_pred_noisy,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        ds_clean.y_test,
        y_pred_clean,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )

    metrics = {
        "noisy_domain": {
            CLASS_NAMES[i]: {
                "precision": float(prec_n[i]),
                "recall": float(rec_n[i]),
                "f1": float(f1_n[i]),
                "support": int(sup_n[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
        "clean_domain": {
            CLASS_NAMES[i]: {
                "precision": float(prec_c[i]),
                "recall": float(rec_c[i]),
                "f1": float(f1_c[i]),
                "support": int(sup_c[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
    }

    out_json = fig_dir / "cnn1d_hyperopt_per_class_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    noisy_acc = float(accuracy_score(ds_noisy.y_test, y_pred_noisy))
    noisy_f1 = float(f1_score(ds_noisy.y_test, y_pred_noisy, average="macro"))
    clean_acc = float(accuracy_score(ds_clean.y_test, y_pred_clean))
    clean_f1 = float(f1_score(ds_clean.y_test, y_pred_clean, average="macro"))

    md_lines = [
        "# Hyperparameter Optimised Model Results",
        "",
        "## Best Hyperparameters",
        "",
        "```json",
        json.dumps(best, indent=2),
        "```",
        "",
        "## Summary",
        "",
        "| Domain | Accuracy | Macro F1 | Epochs Trained |",
        "|---|---:|---:|---:|",
        f"| Noisy test domain | {noisy_acc:.4f} | {noisy_f1:.4f} | {len(hist_noisy['train_loss'])} |",
        f"| Clean test domain | {clean_acc:.4f} | {clean_f1:.4f} | {len(hist_clean['train_loss'])} |",
        "",
        "## Generated Figures",
        "",
        "- cnn1d_hyperopt_noisy_training_curves.png",
        "- cnn1d_hyperopt_clean_training_curves.png",
        "- cnn1d_hyperopt_noisy_domain_cm_test.png",
        "- cnn1d_hyperopt_clean_domain_cm_test.png",
        "- cnn1d_hyperopt_noisy_domain_per_class_metrics_test.png",
        "- cnn1d_hyperopt_clean_domain_per_class_metrics_test.png",
        "",
        "## Per-Class Metrics (Noisy Domain)",
        "",
        *_format_per_class_table(metrics["noisy_domain"]),
        "",
        "## Per-Class Metrics (Clean Domain)",
        "",
        *_format_per_class_table(metrics["clean_domain"]),
        "",
    ]

    out_report = fig_dir / "cnn1d_hyperopt_results_report.md"
    out_report.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved figures to: {fig_dir.as_posix()}")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"- {p.as_posix()}")
    print(f"Saved metrics JSON: {out_json.as_posix()}")
    print(f"Saved report: {out_report.as_posix()}")


if __name__ == "__main__":
    main()
