# Additional Experiments: Running Instructions

Two follow-up experiments that strengthen the dissertation's analysis of the noise memorisation vs accommodation hypothesis.

## Prerequisites

- Python virtual environment already set up (`.venv/`)
- Main experiments have been run at least once (the saved model `results/models/cnn1d_noisy_best.pt` must exist)
- Dataset at `data/eit_dataset.mat`

## Quick Start

```bash
# Run both experiments via CLI
eit experiments additional
```

## Experiment 2: Different Noise Draw Test (Recommended First)

**Runtime:** ~10 seconds (inference only, no training)

**What it does:** Loads the existing noisy-trained CNN (`cnn1d_noisy_best.pt`) and evaluates it on 10 freshly-generated noisy test sets (same noise distribution, different random seeds). If accuracy is stable across draws, the model hasn't memorised specific noise instances.

**Command:**
```bash
eit experiments additional
```

**Expected output:**
```
Original noise test: ~57-76% (depends on saved model's training config)
Alt-draw mean:       similar (± ~1%)
Drop from original:  < 5 percentage points
```

**Interpretation:**
- Drop < 5pp → Evidence AGAINST memorisation (model generalises across noise instances)
- Drop 5–15pp → Partial memorisation of noise statistics
- Drop > 15pp → Evidence FOR memorisation

**Results saved to:** `results/additional_experiments/different_draw/results.json`

### Note on Sanity Check Accuracy

The sanity check may show ~58% rather than the reported 76.1% if the saved model was trained via the `train.py` CLI (default hyperparameters) rather than `run_all_experiments.py` (which uses optimised noisy-training params: dropout=0.4, weight_decay=1e-3, label_smoothing=0.05). **The relative comparison is what matters** — if accuracy is stable across noise draws, the conclusion holds regardless of absolute level.

To get the full 76.1%, re-train the noisy model with the full experiment suite:
```bash
eit experiments run-all
```
Then re-run Experiment 2.

---

## Experiment 1: Fixed-Bias Augmentation

**Runtime:** ~30–60 minutes (trains a new CNN from scratch)

**What it does:** Trains a CNN where noise is sampled ONCE per training sample and held fixed across all epochs. This mimics the deployment scenario where each physical device has persistent per-electrode characteristics (fixed bias, fixed contact impedance).

**Hypothesis:** This bridges the gap between:
- Online augmentation (different noise every epoch → fails at ~20%)
- Fixed noisy dataset (same noise always → succeeds at ~76%)

By assigning persistent noise per sample (but different across samples), the model can learn that noise is a stable per-device property.

**Command:**
```bash
eit experiments additional
```

**Expected output:**
```
[Fixed-Bias] Epoch 10/200 | Train Loss: X.XXXX | Val Loss: X.XXXX | Val Acc: X.XXXX
...
--- Evaluation: Fixed-Bias Model ---
  Clean test accuracy:  XX.X%
  Noisy test accuracy:  XX.X%
  Alt-noise accuracy:   XX.X%
```

**Key comparisons:**
| Condition | Expected Noisy Accuracy |
|-----------|------------------------|
| Online augmentation (baseline) | ~20% |
| Fixed-bias (this experiment) | 20–76% (the gap tells us about noise persistence) |
| Fixed noisy dataset | ~76% |

**Results saved to:**
- `results/additional_experiments/fixed_bias/results.json` — metrics
- `results/additional_experiments/fixed_bias/cnn1d_fixed_bias_best.pt` — model weights
- `results/additional_experiments/fixed_bias/training_history.json` — loss/accuracy curves

---

## How to Use Results in the Dissertation

### Experiment 2 (Different Draw)
Add to §5.2 (Discussion of noise accommodation):
> "To rule out memorisation of specific noise instances, we evaluated the trained model on 10 independently-generated noisy test sets (same parametric distribution, different random seeds). Accuracy remained stable at X.X% ± Y.Y% (original: Z.Z%), confirming the model learns distributional accommodation rather than memorising training-set noise patterns."

### Experiment 1 (Fixed-Bias)
Add to §5.3 or as a new subsection:
> "The fixed-bias augmentation experiment (noise sampled once per training sample, held constant across epochs) achieved X.X% on noisy evaluation, compared to 19.8% for online augmentation and 76.1% for the fixed noisy dataset. This [confirms/suggests] that noise persistence during training is [critical/beneficial] for learning accommodation strategies."

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure the package is installed: `uv pip install -e .` |
| `Model not found` | Run `eit train cnn` first to generate the saved model |
| `CUDA out of memory` | Runs on CPU by default; set `CUDA_VISIBLE_DEVICES=""` to force CPU |
| Experiment 1 too slow | Reduce epochs: edit `config.yaml` → `training.epochs: 100` |
