function dv_noisy = add_noise(dv, params)
%ADD_NOISE Apply physically-motivated noise components to EIT measurements.
%
%   dv_noisy = ADD_NOISE(dv, params) adds configurable noise to the voltage
%   difference vector dv. Each noise component can be independently toggled
%   via the params struct, enabling systematic ablation studies.
%
%   Noise Components:
%     1. Gaussian measurement noise (SNR-parameterised, with baseline floor)
%     2. Electrode contact impedance variation (per-electrode, multiplicative)
%     3. Electrode bias (per-electrode random offset mapped to measurements)
%     4. Quantisation noise (ADC bit-depth)
%
%   Parameters:
%     dv     - (n_meas x 1) clean voltage difference vector
%     params - struct with fields:
%       .gaussian.enabled, .gaussian.snr_db, .gaussian.noise_floor
%       .contact_impedance.enabled, .contact_impedance.std_percent
%       .electrode_bias.enabled, .electrode_bias.max_bias
%       .electrode_bias.n_electrodes
%       .quantisation.enabled, .quantisation.adc_bits, .quantisation.voltage_range
%
%   Returns:
%     dv_noisy - (n_meas x 1) noisy voltage difference vector
%
%   References:
%     [1] Adler & Lionheart (2006) - measurement noise characterisation
%     [2] Vilhunen et al. (2002) - contact impedance variation
%     [3] Boone & Holder (1996) - temporal drift in EIT systems

    arguments
        dv (:,1) double
        params struct
    end

    dv = double(dv);
    dv_noisy = dv;
    n_meas = length(dv);

    % --- 1. Gaussian measurement noise ---
    % Uses SNR-based scaling when signal is present; falls back to a fixed
    % noise floor for zero-signal cases (e.g., "no contact" class) so that
    % the class still has realistic measurement noise.
    if params.gaussian.enabled
        snr_db = params.gaussian.snr_db;
        signal_power = norm(dv);
        noise = randn(n_meas, 1);
        % Zero-signal tolerance rather than an exact ">0" test, for parity with
        % the Python implementation (src/eit_sim2real/data/noise.py). There the
        % clean vector may carry a ~1e-10 float residual from a scaler
        % round-trip, which an exact test would mistake for real signal and so
        % skip the noise floor entirely for the "no contact" class. The margin
        % is wide: the smallest genuine contact signal has norm ~7e-4.
        if signal_power > 1e-6
            scale = signal_power / norm(noise) * 10^(-snr_db / 20);
        else
            % Baseline noise floor for zero-signal measurements
            % Typical EIT systems have ~1e-4 V RMS baseline noise
            scale = params.gaussian.noise_floor / norm(noise) * n_meas;
        end
        dv_noisy = dv_noisy + scale * noise;
    end

    % --- 2. Electrode contact impedance variation ---
    % Applied per-electrode: each electrode has a random impedance factor
    % that affects all measurements involving that electrode. This is
    % physically correct as contact impedance is a property of the
    % electrode-skin interface, not of individual measurements.
    if params.contact_impedance.enabled
        std_frac = params.contact_impedance.std_percent / 100;
        n_elec = params.contact_impedance.n_electrodes;
        meas_per_elec = n_meas / n_elec;

        % Generate per-electrode impedance factors (log-normal)
        elec_factors = exp(std_frac * randn(n_elec, 1));

        % Map to measurements: each electrode is mapped to a contiguous
        % block of (n_meas/n_elec) measurements via REPELEM, so every
        % measurement participating in a given electrode block is multiplied
        % by that electrode's single impedance factor. (Earlier docstrings
        % described this as a 'mean of the two drive electrodes' — that was
        % a description of an alternative scheme that was never implemented;
        % the per-electrode block mapping is what the dissertation §3
        % methodology describes and what the Python noise model mirrors.)
        impedance_factor = repelem(elec_factors, round(meas_per_elec));
        % Trim or pad to match n_meas exactly
        if length(impedance_factor) > n_meas
            impedance_factor = impedance_factor(1:n_meas);
        elseif length(impedance_factor) < n_meas
            impedance_factor(end+1:n_meas) = 1.0;
        end

        dv_noisy = dv_noisy .* impedance_factor;
    end

    % --- 3. Electrode bias ---
    % Per-electrode random offset mapped to the measurement vector.
    % Models systematic positioning errors or gel thickness variation at
    % each electrode site. Each electrode contributes a fixed bias to all
    % measurements it participates in.
    % Ref: [4] Kolehmainen et al. (1997) - 1-2 mm positioning errors
    if params.electrode_bias.enabled
        max_bias = params.electrode_bias.max_bias;
        n_elec = params.electrode_bias.n_electrodes;
        meas_per_elec = n_meas / n_elec;

        % Random per-electrode bias (uniform)
        elec_bias = max_bias * (2 * rand(n_elec, 1) - 1);

        % Map to measurement vector
        bias_vec = repelem(elec_bias, round(meas_per_elec));
        if length(bias_vec) > n_meas
            bias_vec = bias_vec(1:n_meas);
        elseif length(bias_vec) < n_meas
            bias_vec(end+1:n_meas) = 0;
        end

        dv_noisy = dv_noisy + bias_vec;
    end

    % --- 4. Quantisation noise ---
    if params.quantisation.enabled
        n_bits = params.quantisation.adc_bits;
        v_range = params.quantisation.voltage_range;
        lsb = v_range / (2^n_bits);  % Least significant bit
        % Uniform quantisation noise: [-LSB/2, +LSB/2]
        quant_noise = (rand(n_meas, 1) - 0.5) * lsb;
        dv_noisy = dv_noisy + quant_noise;
    end
end