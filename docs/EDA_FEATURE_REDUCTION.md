# EDA Process and Final Pipeline Decision

## Purpose
This document describes the finalized EDA workflow implemented in `notebooks/eda_analysis.ipynb` and the evidence-based preprocessing decisions for the training pipeline.

## Final Pipeline Decision

| Component | Decision | Evidence |
|-----------|----------|----------|
| **Scaler** | RobustScaler | Significantly outperforms StandardScaler (Wilcoxon p<0.001, Cohen's d=2.29) |
| **Feature reduction** | None | Correlation pruning at ρ=0.75 reduces F1 by 3.5% (p<0.001). PCA reduces by 4.5%. |
| **Features** | Full 208 electrodes | The 1D-CNN exploits spatial adjacency in the measurement array |
| **Deduplication** | Not needed | No exact duplicates exist in non-class-0 samples |

## Exported Datasets

| File | Content | Usage |
|------|---------|-------|
| `data/cleaned/eit_cleaned.mat` | Raw 25000×208, full features | **Primary**. Apply RobustScaler in training (fit on train only). |
| `data/cleaned/eit_cleaned_pca.mat` | RobustScaler+PCA (7 components, 95.3% var) | Comparison only. Do NOT re-scale. |
| `data/cleaned/eit_cleaned_lda.mat` | RobustScaler+LDA (4 components) | Comparison only. Do NOT re-scale. |
| `data/cleaned/eit_cleaned_umap.mat` | RobustScaler+UMAP (2 components) | Visualisation only. |
| `data/cleaned/eda_decision_log.json` | Full decision metadata | Reproducibility audit trail. |

## Input Data
- Source: `data/eit_dataset.mat`
- Shape: 25,000 samples × 208 features
- Labels: 5 classes (0-indexed after Python loading)
- Balanced: 5,000 samples per class

## Methodology

### 1. Dataset Integrity Audit
- Class balance: perfectly balanced (5000 per class)
- Exact duplicates: Class 0 only (physically correct — homogeneous medium produces identical EIT measurements)
- Non-class-0 duplicates: **zero** (no deduplication needed)
- Near-duplicate analysis confirms clean data generation

### 2. Distribution Diagnostics
- Shapiro-Wilk tests on feature subsets confirm non-Gaussian distributions across all representations
- Mean absolute skewness and excess kurtosis are similar across Standard, Robust, and MinMax scalings
- Visual inspection shows heavy-tailed features that benefit from robust scaling

### 3. Leakage-Safe Pipeline Comparison (Repeated Stratified CV)
Method: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` = 15 paired evaluations

| Pipeline | Macro-F1 (mean±std) | Balanced Accuracy |
|----------|--------------------:|------------------:|
| **robust_lr** | **0.718 ± 0.004** | **0.721** |
| standard_lr | 0.713 ± 0.004 | 0.717 |
| corr090_f_lr | 0.704 ± 0.004 | 0.707 |
| lda_classifier | 0.683 ± 0.110 | 0.686 |
| corr075_f_lr | 0.683 ± 0.007 | 0.686 |
| pca95_lr | 0.673 ± 0.005 | 0.676 |
| corr075_pca95_lr | 0.672 ± 0.007 | 0.675 |

### 4. Statistical Significance (Wilcoxon Signed-Rank, Paired)

| Comparison | Δ(F1) | Cohen's d | p-value | Significant |
|------------|-------:|----------:|--------:|:-----------:|
| robust vs standard | +0.005 | 2.29 | <0.001 | ✓ |
| robust vs corr090 | +0.014 | 3.86 | <0.001 | ✓ |
| standard vs corr090 | +0.009 | 2.94 | <0.001 | ✓ |

**Key insight**: Despite the small absolute difference (0.5%), RobustScaler is *consistently* better across all 15 folds (hence the large effect size). This makes it the scientifically defensible default.

### 5. Why Feature Reduction Hurts

The 208 EIT features represent voltage measurements from a spatial electrode array. Adjacent electrodes share spatial information that the 1D-CNN exploits via convolutional filters. Removing "correlated" features destroys this spatial structure. For linear models, the small redundancy penalty is outweighed by the information lost.

## Training Pipeline Integration

The training scripts (`train.py`, `evaluate.py`) now use:
```python
prepare_splits(X, y, scaler_type="robust")  # config.yaml: data.scaler = "robust"
```

For PCA/LDA variant experiments, load the pre-transformed datasets with `normalize=False` to avoid double-scaling.

## Reproducibility
- All decisions recorded in `eda_decision_log.json`
- Random seed: 42
- Python packages: scikit-learn, scipy, numpy (versions in `pyproject.toml`)

## Notes on Duplicates and Data Generation
The prior dataset-generation issue (contact mask occasionally empty for contact classes) could create degenerate zero-difference samples. That issue has been fixed in MATLAB generation code; duplicate audits should always be rerun after dataset regeneration.

## Reproducibility
- Notebook: `notebooks/eda_analysis.ipynb`
- Decision log: `data/cleaned/eda_decision_log.json`
- Fixed seed: `SEED = 42`

If the raw dataset changes, rerun the notebook and regenerate all cleaned outputs and decision logs.
