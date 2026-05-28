# Raw Data Creation and Literature Basis

## Purpose
This document explains how the raw EIT dataset was generated in this project, what assumptions were made, and which literature supports those choices.

Primary implementation path:
- MATLAB pipeline in `matlab/main.m`
- Sample synthesis in `matlab/generate_sample.m`
- Mesh setup in `matlab/create_mesh.m`
- Noise parameter loading in `matlab/noise_model/load_noise_params.m`
- Noise defaults in `matlab/configs/noise_params.yaml`

## Dataset Generation Pipeline

### 1. Simulation setup
The dataset is generated using EIDORS forward simulation (not hardware capture).

Configured defaults:
- Total samples: 25,000
- Class balance: 5 classes x 5,000 each
- Geometry: `2d_circle` (default)
- Electrodes: 16
- Measurements: 208 per sample (adjacent stimulation/measurement pattern)
- Random seed: 42 (applied after EIDORS startup to ensure the seed controls data generation, not setup)

Rationale:
- EIDORS is a standard and well-documented EIT simulation framework for reproducible methodology development (Adler and Lionheart, 2006).
- 16-electrode adjacent pattern is a common practical compromise between spatial coverage and complexity in EIT sensing systems.
- `rng(seed)` is placed immediately before the generation loop rather than at script start; EIDORS' startup script modifies MATLAB's random state internally, so seeding before startup would be overwritten.

### 2. Baseline and perturbation image construction
For each sample:
1. Build homogeneous baseline image (`sigma = 1.0`).
2. Generate class-specific contact parameters.
3. Apply local conductivity perturbation in a circular region (except no-contact class).
4. Run forward solve to obtain `vi.meas`.
5. Compute differential measurement vector:
   - `dv_clean = vi.meas - vh.meas`
6. Apply noise chain to obtain `dv_noisy`.

This differential formulation follows common EIT practice where change signals are used for sensing tasks.

**Contact mask nearest-element fallback**: The circular mask `dist < radius` is applied to element centroids on a discrete FEM mesh. For small-radius contacts (especially `point`, r = 0.02–0.05) on a `~1024-element` mesh (centroid spacing ~0.05–0.08), the contact circle can contain zero element centroids. When this occurs, the forward solve produces the same result as the homogeneous baseline (`dv_clean = 0`), creating a degenerate zero-vector that is indistinguishable from the no-contact class. The fix in `generate_sample.m` ensures that if the mask is empty, the single nearest element centroid is included. This is physically consistent: a point load always perturbs the element it is located on (Kim et al., 2019).

### 3. Touch class parameterisation
Five touch classes are synthesised:
- No contact
- Light touch
- Firm press
- Point contact
- Distributed contact

Class parameters (from `generate_sample.m`) are sampled stochastically per sample.

| Class | Radius range | Conductivity range | Intended effect |
|---|---:|---:|---|
| No contact | 0 | 1.00 | Reference baseline |
| Light | 0.06 to 0.10 | 0.85 to 0.95 | Mild local decrease |
| Firm | 0.08 to 0.12 | 0.55 to 0.75 | Stronger decrease |
| Point | 0.02 to 0.05 | 0.35 to 0.55 | Small-area, high local contrast |
| Distributed | 0.15 to 0.25 | 0.80 to 0.92 | Large-area, lower per-element contrast |

Additional spatial rule:
- Contact center is sampled within 40% of mesh bounds to reduce extreme boundary-sensitivity effects dominating class variance.

Interpretation:
- These are physics-informed priors designed to produce plausible class structure and separability for robustness studies, not direct fitted constants from one hardware rig.

### 4. Noise injection
`dv_clean` is transformed into `dv_noisy` using a 4-component model:
1. Gaussian measurement noise
2. Contact impedance variation (per-electrode multiplicative)
3. Electrode positioning bias (per-electrode additive)
4. Quantisation noise

Default values are loaded from `matlab/configs/noise_params.yaml`.

Key defaults:
- Gaussian SNR: 40 dB
- Gaussian noise floor: 1e-4 V RMS
- Contact impedance std: 10%
- Electrode bias max: 0.02
- ADC bits: 16
- ADC full-scale range: 1.0 V

See `docs/NOISE_MODEL.md` for full derivations and implementation details.

### 5. Outputs
MATLAB generation writes:
- `data/eit_dataset.mat` (full metadata, clean and noisy vectors)
- `data/eit_dataset_numpy.mat` (Python-friendly arrays)

Stored fields include:
- `dataset_X_clean`
- `dataset_X_noisy`
- `dataset_y`
- `dataset_metadata`
- `config`
- `noise_params`

## Literature Support for Design Choices

## EIT simulation and reproducibility
- Adler and Lionheart (2006): establishes EIDORS as an extensible EIT simulation platform used for reproducible studies.

## Electrode/interface and error sensitivity
- Boone and Holder (1996): demonstrates skin/electrode impedance effects on EIT image quality and variability.
- Vilhunen et al. (2002): supports modeling contact impedance uncertainty as a significant EIT error source.
- Kolehmainen et al. (1997): quantifies electrode-position-related EIT errors.

These motivate including contact and bias terms in the synthetic corruption model, not only additive Gaussian noise.

## E-skin and prosthetic context
- Chortos, Liu and Bao (2016): reviews prosthetic e-skin design pressures and practical constraints.
- Lee et al. (2020): highlights ML + e-skin integration challenges in realistic deployment conditions.
- Yao, Chen and Gao (2022): demonstrates impedance-tomography-based tactile skin design relevance.

These support the project-level goal: training and validating under realistic corruption rather than idealized-only data.

## Hydrogel and tactile EIT relevance
- Zhang et al. (2022): EIT-based hydrogel tactile sensing using reconstruction algorithms, supporting hydrogel-EIT feasibility assumptions.
- Literature notes and project comments indicate hydrogel/mechano-electrical behavior as a key motivation for sign and magnitude choices in class perturbations.

## Why synthetic raw data is still useful here
The project is explicitly scoped as simulation-first and robustness-focused:
- Enables controlled perturbation and ablation at scale.
- Keeps clean/noisy paired samples aligned for causal comparison.
- Supports repeatable experimentation with fixed seeds and versioned configs.

This does not replace hardware validation; it provides a controlled experimental substrate for sim-to-real robustness analysis.

## Assumptions and limitations
- Raw data is simulated, not acquired from a physical prosthetic sleeve.
- Touch class ranges are calibrated for plausible class structure and separability, not direct one-to-one patient-specific biomechanics.
- Default geometry is 2D; 3D exists as an optional extension path.
- Noise components are physically motivated approximations of a broader real signal chain.

## References Used in This Document
- Adler, A. and Lionheart, W.R.B. (2006) Uses and abuses of EIDORS.
- Boone, K.G. and Holder, D.S. (1996) Effect of skin impedance on EIT quality.
- Vilhunen, T. et al. (2002) Contact impedance reconstruction theory.
- Kolehmainen, V. et al. (1997) Error assessment in static EIT.
- Chortos, A., Liu, J. and Bao, Z. (2016) Pursuing prosthetic electronic skin.
- Lee, Y. et al. (2020) Electronic skins and machine learning for intelligent soft robots.
- Yao, J., Chen, H. and Gao, Y. (2022) Biomimetic robot skin using impedance/acoustic tomography.
- Zhang, Y. et al. (2022) Hydrogel EIT e-skin for multi-touch sensing.
