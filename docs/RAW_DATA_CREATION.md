# EIT Dataset Generation — Specification and Literature Basis

## Overview

This document specifies the dataset generation procedure employed in this project,
documents the assumptions underlying each design choice, and provides the supporting
literature references.

**Implementation files**:
- MATLAB pipeline: `matlab/main.m`
- Sample synthesis: `matlab/generate_sample.m`
- Mesh construction: `matlab/create_mesh.m`
- Noise parameter loading: `matlab/noise_model/load_noise_params.m`
- Noise configuration defaults: `matlab/configs/noise_params.yaml`

## Dataset Generation Pipeline

### 1. Simulation Configuration

The dataset is generated via EIDORS forward simulation with the following
configuration:

**Configured defaults**:
- Total samples: 25,000
- Class balance: 5 classes x 5,000 each
- Geometry: `2d_circle` 
- Electrodes: 16
- Measurements: 208 per sample (adjacent stimulation/measurement pattern)
- Random seed: 42 (applied after EIDORS startup to ensure the seed controls data generation, not setup)

**Rationale**:
- EIDORS is an established and well-documented EIT simulation framework suitable
  for reproducible methodology development (Adler & Lionheart, 2006).
- The 16-electrode adjacent stimulation/measurement pattern represents a standard
  practical compromise between spatial coverage and system complexity in EIT
  sensing (Holder, 2004).
- The random seed is applied immediately before the generation loop rather than at
  script start, as the EIDORS startup routine modifies MATLAB's random number
  generator state internally.

### 2. Baseline and Perturbation Image Construction

For each sample, the following procedure is executed:
1. Build homogeneous baseline image (`sigma = 1.0`).
2. Generate class-specific contact parameters.
3. Apply local conductivity perturbation in a circular region (except no-contact class).
4. Run forward solve to obtain `vi.meas`.
5. Compute differential measurement vector:
   - `dv_clean = vi.meas - vh.meas`
6. Apply noise chain to obtain `dv_noisy`.

This differential formulation follows standard EIT practice in which change
signals relative to a known baseline are used for sensing tasks (Holder, 2004).

**Contact mask nearest-element fallback**: The circular mask `dist < radius` is
applied to element centroids on the discrete FEM mesh. For small-radius contacts
(particularly the `point` class, $r = 0.02$–$0.05$) on a mesh with approximately
1024 elements (centroid spacing ~0.05–0.08), the contact circle may contain zero
element centroids. When this occurs, the forward solve produces the same result as
the homogeneous baseline ($\Delta v_{\text{clean}} = 0$), creating a degenerate
zero-vector indistinguishable from the no-contact class. The implementation in
`generate_sample.m` addresses this by including the single nearest element centroid
when the mask is empty. This is physically consistent with the discrete FEM
interpretation of a point load: the element directly underlying the contact point
is always perturbed (Kim et al., 2019).

### 3. Touch Class Parameterisation

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

**Spatial constraint**: The contact centre is sampled within 40% of the mesh
bounds to mitigate extreme boundary-sensitivity effects that would otherwise
dominate within-class variance.

**Interpretation**: These parameters constitute physics-informed priors designed to
produce plausible class structure and separability for robustness studies. They are
not directly fitted constants from a specific hardware system.

### 4. Noise Injection

The clean differential voltage vector is transformed into its noisy counterpart
using the 4-component physically-motivated noise model:
1. Gaussian measurement noise
2. Contact impedance variation (per-electrode multiplicative)
3. Electrode positioning bias (per-electrode additive)
4. Quantisation noise

Default values are loaded from `matlab/configs/noise_params.yaml`.

**Default parameter values**:
- Gaussian SNR: 40 dB
- Gaussian noise floor: $1 \times 10^{-4}$ V RMS
- Contact impedance std: 10%
- Electrode bias max: 0.02
- ADC bits: 16
- ADC full-scale range: 1.0 V

See `docs/NOISE_MODEL.md` for full derivations and implementation details.

### 5. Output Artefacts

The MATLAB generation pipeline produces:
- `data/eit_dataset.mat` (full metadata, clean and noisy vectors)
- `data/eit_dataset_numpy.mat` (Python-friendly arrays)

**Stored fields**:
- `dataset_X_clean`
- `dataset_X_noisy`
- `dataset_y`
- `dataset_metadata`
- `config`
- `noise_params`

## Literature Support for Design Choices

### EIT Simulation and Reproducibility
- Adler & Lionheart (2006): establishes EIDORS as an extensible EIT simulation
  platform suitable for reproducible studies.

### Electrode Interface and Error Sensitivity
- Boone & Holder (1996): demonstrates skin–electrode impedance effects on EIT
  image quality and variability.
- Vilhunen et al. (2002): supports modelling contact impedance uncertainty as a
  significant EIT error source.
- Kolehmainen et al. (1997): quantifies electrode-position-related EIT errors.

These references motivate the inclusion of contact impedance and bias terms in the
synthetic corruption model, beyond additive Gaussian noise alone.

### Electronic Skin and Prosthetic Context
- Chortos, Liu & Bao (2016): reviews prosthetic e-skin design constraints.
- Lee et al. (2020): identifies ML and e-skin integration challenges under
  realistic deployment conditions.
- Yao, Chen & Gao (2022): demonstrates impedance-tomography-based tactile skin
  design relevance.

These sources support the project-level objective of training and validating under
realistic corruption rather than idealised conditions only.

### Hydrogel and Tactile EIT Relevance
- Zhang et al. (2022): EIT-based hydrogel tactile sensing using reconstruction
  algorithms, supporting the hydrogel–EIT feasibility assumptions underlying the
  material model.
- The hydrogel mechano-electrical behaviour (positive piezoresistive response)
  motivates the sign and magnitude choices in the class conductivity
  perturbations.

## Justification for Synthetic Data

The project is explicitly scoped as simulation-first and robustness-focused. The
synthetic approach:
- Enables controlled perturbation and ablation at scale.
- Maintains paired clean/noisy samples for causal comparison.
- Supports repeatable experimentation with fixed seeds and versioned configurations.

This does not replace hardware validation; it provides a controlled experimental
substrate for sim-to-real robustness analysis.

## Assumptions and Limitations

- All data are simulated, not acquired from a physical prosthetic system.
- Touch class parameter ranges are calibrated for plausible class structure and
  separability, not directly fitted to patient-specific biomechanics.
- The default geometry is 2D; 3D cylindrical geometry exists as an extension path.
- Noise components are physically motivated first-order approximations of a
  broader real signal chain.

## References

1. Adler, A. & Lionheart, W.R.B. (2006). Uses and abuses of EIDORS: an extensible software base for EIT. *Physiological Measurement*, 27(5), S25–S42.
2. Boone, K. & Holder, D. (1996). Effect of skin impedance on image quality and variability in electrical impedance tomography. *Medical & Biological Engineering & Computing*, 34(5), 351–354.
3. Chortos, A., Liu, J. & Bao, Z. (2016). Pursuing prosthetic electronic skin. *Nature Materials*, 15(9), 937–950.
4. Holder, D.S. (Ed.) (2004). *Electrical Impedance Tomography: Methods, History and Applications*. Institute of Physics Publishing.
5. Kim, B. et al. (2019). Deep learning-based electrical impedance tomography. *Physiological Measurement*, 40(5), 054001.
6. Kolehmainen, V. et al. (1997). Assessment of errors in static electrical impedance tomography with adjacent and trigonometric current patterns. *Physiological Measurement*, 18(4), 289–303.
7. Lee, Y. et al. (2020). Skin-like electronics for perception and interaction. *Advanced Functional Materials*, 30(36), 2000540.
8. Vilhunen, T. et al. (2002). Simultaneous reconstruction of electrode contact impedances and internal electrical properties: I. Theory. *Measurement Science and Technology*, 13(12), 1848–1854.
9. Yao, J., Chen, H. & Gao, Y. (2022). A biomimetic elastomeric robot skin using electrical impedance and acoustic tomography for tactile sensing. *Advanced Intelligent Systems*, 4(12), 2200162.
10. Zhang, Y. et al. (2022). Hydrogel-based EIT e-skin for multi-touch sensing. *Soft Robotics*, 9(4), 745–756.
