"""Three follow-up experiments addressing weaknesses found during
source-verification of the bibliography.

1. bias_correlation sweep
   Kolehmainen et al. (1997) and Boone & Holder (1996) both report that
   difference imaging largely cancels electrode error that is *constant*
   between the reference and measurement frames. The main results apply
   electrode bias directly to dv, implicitly assuming zero correlation
   between frames. This sweeps that assumption.

2. permutation ablation
   Tests the claim that the CNN advantage comes from the block structure of
   the corruption rather than from capacity. A fixed permutation of the 208
   measurement indices, applied identically to train and test AFTER noise
   injection, preserves information content while destroying adjacency.

3. no-contact exclusion
   The no-contact class is an exact zero vector by construction, so the
   measured value of the Gaussian component may be an artefact of rescuing a
   degenerate class. Re-runs the key subset ablation on the four contact
   classes only, under deployment-realistic evaluation.

Protocol matches the main grid: stratified 70/15/15, scaler fitted on the
noisy training split only, CNN with the noise-trained regularisation preset.

Outputs: results/additional_experiments/reviewer/reviewer_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from eit_sim2real.data.load_dataset import load_mat_dataset
from eit_sim2real.data.noise import NoiseConfig, apply_noise_batch_vectorised
from eit_sim2real.models.baselines import get_baseline
from eit_sim2real.train import train_cnn

DATA = Path("data/eit_dataset_numpy.mat")
OUT = Path("results/additional_experiments/reviewer")
SEEDS = (42, 43, 44, 45, 46)


# ---------------------------------------------------------------- helpers
def split_clean(X: np.ndarray, y: np.ndarray, seed: int):
    """Stratified 70/15/15 split on CLEAN data; noise applied afterwards."""
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, random_state=seed, stratify=y_tv
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def noise(X: np.ndarray, cfg: NoiseConfig, seed: int) -> np.ndarray:
    return apply_noise_batch_vectorised(X, cfg, rng=np.random.default_rng(seed))


def fit_scale(X_tr, X_va, X_te):
    sc = RobustScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)


def cnn_acc(X_tr, y_tr, X_va, y_va, X_te, y_te, seed, n_classes=5) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, _ = train_cnn(
        X_tr, y_tr, X_va, y_va, n_classes=n_classes,
        epochs=200, dropout=0.4, label_smoothing=0.05,
    )
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_te, dtype=torch.float32).unsqueeze(1).to(dev)
        pred = model(xb).argmax(1).cpu().numpy()
    return float((pred == y_te).mean())


def baseline_accs(X_tr, y_tr, X_te, y_te, seed) -> dict:
    out = {}
    for name in ("svm", "random_forest", "mlp"):
        try:
            clf = get_baseline(name, random_state=seed)
            clf.fit(X_tr, y_tr)
            out[name] = float((clf.predict(X_te) == y_te).mean())
        except Exception as exc:
            out[name] = f"ERROR: {exc}"
    return out


# ------------------------------------------------- 1. correlation sweep
def run_bias_correlation(rhos=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0)) -> dict:
    print("\n=== experiment 1: bias correlation sweep ===", flush=True)
    X, y = load_mat_dataset(DATA, use_noisy=False)
    res = {}
    for rho in rhos:
        accs = []
        for seed in SEEDS:
            cfg = NoiseConfig(bias_correlation=rho)
            Xtr, Xva, Xte, ytr, yva, yte = split_clean(X, y, seed)
            Xtr, Xva, Xte = (noise(Xtr, cfg, seed), noise(Xva, cfg, seed + 1),
                             noise(Xte, cfg, seed + 2))
            Xtr, Xva, Xte = fit_scale(Xtr, Xva, Xte)
            a = cnn_acc(Xtr, ytr, Xva, yva, Xte, yte, seed)
            accs.append(a)
            print(f"  rho={rho:<5} seed={seed}  acc={a*100:6.2f}", flush=True)
        res[str(rho)] = {"accuracies": accs, "mean": float(np.mean(accs)),
                         "std": float(np.std(accs))}
        print(f"  rho={rho:<5} MEAN {np.mean(accs)*100:6.2f} "
              f"+/- {np.std(accs)*100:.2f}", flush=True)
    return res


# ------------------------------------------------- 2. permutation ablation
def run_permutation() -> dict:
    print("\n=== experiment 2: permutation ablation ===", flush=True)
    X, y = load_mat_dataset(DATA, use_noisy=False)
    cfg = NoiseConfig()
    perm = np.random.default_rng(12345).permutation(X.shape[1])
    res = {}
    for tag, do_perm in (("original", False), ("permuted", True)):
        accs, bl = [], None
        for seed in SEEDS:
            Xtr, Xva, Xte, ytr, yva, yte = split_clean(X, y, seed)
            Xtr, Xva, Xte = (noise(Xtr, cfg, seed), noise(Xva, cfg, seed + 1),
                             noise(Xte, cfg, seed + 2))
            if do_perm:  # after noise: corruption stays block-structured
                Xtr, Xva, Xte = Xtr[:, perm], Xva[:, perm], Xte[:, perm]
            Xtr, Xva, Xte = fit_scale(Xtr, Xva, Xte)
            a = cnn_acc(Xtr, ytr, Xva, yva, Xte, yte, seed)
            accs.append(a)
            print(f"  {tag:<9} seed={seed}  CNN={a*100:6.2f}", flush=True)
            if bl is None:
                bl = baseline_accs(Xtr, ytr, Xte, yte, seed)
                print(f"  {tag:<9} baselines="
                      f"{ {k: (round(v*100,2) if isinstance(v,float) else v) for k,v in bl.items()} }",
                      flush=True)
        res[tag] = {"cnn_accuracies": accs, "cnn_mean": float(np.mean(accs)),
                    "cnn_std": float(np.std(accs)), "baselines": bl}
    return res


# ------------------------------------------------- 3. no-contact exclusion
def run_no_contact_exclusion() -> dict:
    print("\n=== experiment 3: no-contact exclusion ===", flush=True)
    X, y = load_mat_dataset(DATA, use_noisy=False)
    m = y != 0
    Xc, yc = X[m], (y[m] - 1)
    print(f"  retained {int(m.sum())}/{len(y)} samples, "
          f"{len(np.unique(yc))} classes", flush=True)

    subsets = {
        "gaussian_only": NoiseConfig(contact_impedance_enabled=False,
                                     electrode_bias_enabled=False,
                                     quantisation_enabled=False),
        "bias_only": NoiseConfig(gaussian_enabled=False,
                                 contact_impedance_enabled=False,
                                 quantisation_enabled=False),
        "gaussian_bias": NoiseConfig(contact_impedance_enabled=False,
                                     quantisation_enabled=False),
        "full_four": NoiseConfig(),
    }
    full = NoiseConfig()
    res = {}
    for name, tcfg in subsets.items():
        accs = []
        for seed in SEEDS:
            Xtr, Xva, Xte, ytr, yva, yte = split_clean(Xc, yc, seed)
            # train on ablated corruption, evaluate on the FULL corruption
            Xtr_n, Xva_n = noise(Xtr, tcfg, seed), noise(Xva, tcfg, seed + 1)
            Xte_n = noise(Xte, full, seed + 2)
            Xtr_s, Xva_s, Xte_s = fit_scale(Xtr_n, Xva_n, Xte_n)
            a = cnn_acc(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, seed, n_classes=4)
            accs.append(a)
            print(f"  {name:<14} seed={seed}  deployment={a*100:6.2f}", flush=True)
        res[name] = {"accuracies": accs, "mean": float(np.mean(accs)),
                     "std": float(np.std(accs))}
        print(f"  {name:<14} MEAN {np.mean(accs)*100:6.2f} "
              f"+/- {np.std(accs)*100:.2f}", flush=True)
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for name, fn in (("permutation", run_permutation),
                     ("bias_correlation", run_bias_correlation),
                     ("no_contact_exclusion", run_no_contact_exclusion)):
        try:
            all_res[name] = fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            all_res[name] = {"ERROR": str(exc)}
        (OUT / "reviewer_results.json").write_text(
            json.dumps(all_res, indent=2), encoding="utf-8")
    print("\nDONE ->", OUT / "reviewer_results.json", flush=True)


if __name__ == "__main__":
    main()
