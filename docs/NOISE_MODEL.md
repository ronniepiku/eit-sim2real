# EIT Touch Classification — Noise Model Specification

## Overview

This document details the physical justification, mathematical formulation, and
implementation of the 4-component noise model used in the EIT simulation pipeline.

The noise model is the primary methodological contribution of this project. It enables
systematic ablation studies to determine which noise sources most affect classifier
robustness during simulation-to-reality transfer.

## Design Principles

1. **Physics-grounded**: Each component models a distinct, documented physical degradation source
2. **Independently configurable**: Components can be toggled on/off for ablation studies
3. **Literature-parameterised**: All default values derive from published EIT hardware characterisations
4. **Per-electrode structure**: Contact impedance and bias are applied per-electrode (not per-measurement), reflecting that these are properties of the electrode–substrate interface
5. **Fixed application order**: Components are applied sequentially (Gaussian → Contact Impedance → Bias → Quantisation), reflecting the physical signal chain

## Component Specifications

### 1. Gaussian Measurement Noise

**Physical origin**: Thermal noise in electrodes, amplifier noise, and electromagnetic interference in the analogue measurement chain.

**Mathematical formulation**:
```
Δv' = Δv + n
n ~ N(0, σ_n² I)
σ_n = (||Δv||₂ / ||n̂||₂) × 10^(-SNR/20)   when ||Δv|| > 0
σ_n = noise_floor × n_meas / ||n̂||₂          when ||Δv|| = 0
```

where `n̂ ~ N(0, I)` is a unit-variance noise realisation.

**Parameters**:
| Parameter | Default | Range | Source |
|-----------|---------|-------|--------|
| SNR (dB) | 40 | 30–80 | Adler & Lionheart (2006) |
| noise_floor (V RMS) | 1×10⁻⁴ | — | Typical EIT baseline noise |

**Notes**: The signal-dependent formulation ensures noise scales proportionally with
measurement amplitude. A fixed noise floor is applied for zero-signal measurements
(e.g., "no contact" class) to ensure realistic baseline noise. The 40 dB default
represents the lower bound for wearable/portable hardware; laboratory systems
typically achieve 60–80 dB.

### 2. Electrode Contact Impedance Variation

**Physical origin**: Imperfect and spatially varying electrode–substrate contact due to
manufacturing variability, gel degradation, surface contaminants, or mechanical loosening.

**Mathematical formulation**:
```
Δv'_i = Δv_i × z_e(i)
z_e = exp(ε_e)
ε_e ~ N(0, σ_z²)
```

where `e(i)` maps measurement `i` to the electrode responsible for that measurement
block. Each electrode has a single impedance factor applied to all measurements it
participates in.

**Parameters**:
| Parameter | Default | Range | Source |
|-----------|---------|-------|--------|
| σ_z (std as fraction) | 0.10 (10%) | 0.05–0.20 | Vilhunen et al. (2002) |
| n_electrodes | 16 | — | System configuration |

**Notes**: The log-normal model ensures z > 0 (impedance is always positive) and
produces asymmetric variation. Applied per-electrode (not per-measurement) as contact
impedance is a property of the electrode–skin interface, not of individual voltage
readings.

### 3. Electrode Positioning Bias

**Physical origin**: Systematic measurement error from imprecise electrode placement,
manufacturing tolerances, or arm geometry variation between subjects.

**Mathematical formulation**:
```
Δv'_i = Δv_i + b_e(i)
b_e ~ U(-b_max, +b_max)
```

where `e(i)` maps measurement `i` to the electrode responsible for that measurement
block. Each electrode has a random fixed bias.

**Parameters**:
| Parameter | Default | Range | Source |
|-----------|---------|-------|--------|
| b_max | 0.02 | 0.01–0.05 | Kolehmainen et al. (1997) |
| n_electrodes | 16 | — | System configuration |

**Notes**: Applied per-electrode with random offsets (rather than a linear gradient
across measurements). This correctly models that each electrode site has its own
positioning error independent of others. The per-electrode structure produces
block-structured bias in the measurement vector.

### 4. Quantisation Noise

**Physical origin**: Finite resolution of the analogue-to-digital converter (ADC).

**Mathematical formulation**:
```
Δv'_i = Δv_i + q_i
q_i ~ U(-LSB/2, +LSB/2)
LSB = V_range / 2^n_bits
```

**Parameters**:
| Parameter | Default | Range | Source |
|-----------|---------|-------|--------|
| n_bits | 16 | 12–24 | EIT hardware specs |
| V_range (V) | 1.0 | 0.5–5.0 | Typical ADC full-scale |

**Notes**: At 16-bit resolution, LSB ≈ 15.3 µV — typically the smallest component.
Becomes significant only for low-resolution portable systems or in severity sweeps.

## Parameter Summary

| Component | Type | Parameter | Default | MATLAB field |
|-----------|------|-----------|---------|--------------|
| Gaussian | Additive | snr_db | 40.0 | `params.gaussian.snr_db` |
| Gaussian | — | noise_floor | 1e-4 | `params.gaussian.noise_floor` |
| Contact impedance | Multiplicative (per-electrode) | std_percent | 10.0 | `params.contact_impedance.std_percent` |
| Bias | Additive (per-electrode) | max_bias | 0.02 | `params.electrode_bias.max_bias` |
| Quantisation | Additive (uniform) | adc_bits | 16 | `params.quantisation.adc_bits` |
| Quantisation | — | voltage_range | 1.0 | `params.quantisation.voltage_range` |

## Ablation Study Design

The noise model enables three ablation configurations:

### Single-Component (4 experiments)
Each component enabled individually — reveals standalone contribution:
- `only_gaussian`, `only_contact_impedance`, `only_electrode_bias`, `only_quantisation`

### Leave-One-Out (4 experiments)
All components except one — reveals marginal value when others present:
- `without_gaussian`, `without_contact_impedance`, `without_electrode_bias`, `without_quantisation`

### Severity Sweep (6 levels)
All parameters scaled by multiplier {0.5×, 1.0×, 1.5×, 2.0×, 2.5×, 3.0×}:
- Tests generalisation beyond training-time noise intensity
- Produces robustness curves (accuracy vs. severity)

## Configuration

Parameters are specified in [`matlab/configs/noise_params.yaml`](../matlab/configs/noise_params.yaml).
Each component has an `enabled` toggle for ablation control.

To generate a clean dataset:
```yaml
enabled: false  # Master toggle disables all components
```

To run single-component ablation:
```yaml
enabled: true
gaussian:
  enabled: true
contact_impedance:
  enabled: false
electrode_bias:
  enabled: false
quantisation:
  enabled: false
```

## Implementation

- **MATLAB**: [`matlab/noise_model/add_noise.m`](../matlab/noise_model/add_noise.m) — applies noise during dataset generation
- **Python** (severity sweep): [`python/evaluate.py`](../python/evaluate.py) — scales noise perturbation at evaluation time

## References

1. Adler, A. & Lionheart, W.R.B. (2006). Uses and abuses of EIDORS: an extensible software base for EIT. *Physiological Measurement*, 27(5), S25–S42.
2. Vilhunen, T. et al. (2002). Simultaneous reconstruction of electrode contact impedances and internal electrical properties: I. Theory. *Measurement Science and Technology*, 13(12), 1848–1854.
3. Boone, K. & Holder, D. (1996). Effect of skin impedance on image quality and variability in electrical impedance tomography. *Medical & Biological Engineering & Computing*, 34(5), 351–354.
4. Kolehmainen, V. et al. (1997). Assessment of errors in static electrical impedance tomography with adjacent and trigonometric current patterns. *Physiological Measurement*, 18(4), 289–303.
5. Hendrycks, D. & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *ICLR*.
