# EIT Touch Classification - Noise Model Derivation

## Overview

This document details the physical justification for each noise component
in the EIT simulation pipeline.

## 1. Gaussian Measurement Noise

**Physical origin**: Electronic noise in the measurement circuitry (amplifiers, ADCs),
electromagnetic interference, and thermal noise in conductors.

**Parameterisation**: Signal-to-noise ratio (SNR) in decibels.
- Laboratory-grade EIT systems: 40–80 dB (Adler & Lionheart, 2006)
- Portable/wearable EIT systems: 20–40 dB
- Default: 40 dB (conservative estimate for wearable e-skin)

**Implementation**: Additive white Gaussian noise scaled to achieve target SNR
relative to the signal norm.

## 2. Electrode Contact Impedance Variation

**Physical origin**: Imperfect electrode-skin coupling due to gel degradation,
hair, sweat, skin irregularities, or manufacturing variability.

**Parameterisation**: Per-electrode multiplicative factor drawn from log-normal
distribution with specified standard deviation.
- Reported variation: 5–20% in clinical EIT (Vilhunen et al., 2002)
- Default: 10% standard deviation

**Implementation**: Each measurement is multiplied by a random factor
drawn from LogNormal(0, σ²) where σ corresponds to the desired percentage variation.

## 3. Systematic Temporal Drift

**Physical origin**: Slow changes in electrode-skin interface (gel drying),
temperature fluctuations, and electronic component aging.

**Parameterisation**: Random walk with bounded magnitude.
- Reported drift: 0.1–1% per minute (Boone & Holder, 1996)
- Default: 0.001 per sample increment, maximum 0.05

**Implementation**: Cumulative sum of small random increments, clipped
to maximum magnitude.

## 4. Electrode Positioning Bias

**Physical origin**: Manufacturing tolerances in electrode placement,
arm geometry variation between subjects.

**Parameterisation**: Linear gradient bias across electrode array.
- Reported positioning error: 1–2 mm (Kolehmainen et al., 1997)
- Default: ±0.02 linear gradient

**Implementation**: Linear ramp from -max_bias to +max_bias across measurements.

## 5. Quantisation Noise

**Physical origin**: Finite resolution of analog-to-digital converter.

**Parameterisation**: ADC bit depth and voltage range.
- Typical EIT hardware: 16–24 bit ADC
- Default: 16-bit, 1V range → LSB = 15.3 µV

**Implementation**: Uniform noise in range [-LSB/2, +LSB/2].

## References

1. Adler, A. & Lionheart, W.R.B. (2006). Uses and abuses of EIDORS: an extensible
   software base for EIT. *Physiological Measurement*, 27(5), S25.
2. Vilhunen, T. et al. (2002). Simultaneous reconstruction of electrode contact
   impedances and internal electrical properties. *Inverse Problems*, 18(5), 1319.
3. Boone, K. & Holder, D. (1996). Effect of skin impedance on image quality and
   variability in electrical impedance tomography. *Medical & Biological Engineering
   & Computing*, 34(5), 351-354.
4. Kolehmainen, V. et al. (1997). Assessment of errors in static electrical impedance
   tomography with adjacent and trigonometric current patterns. *Physiological
   Measurement*, 18(4), 289.
