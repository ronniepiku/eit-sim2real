function dv_noisy = add_noise(dv, params)
%ADD_NOISE Apply physically-motivated noise components to EIT measurements.
%
%   dv_noisy = ADD_NOISE(dv, params) adds configurable noise to the voltage
%   difference vector dv. Each noise component can be independently toggled
%   via the params struct, enabling systematic ablation studies.
%
%   Noise Components:
%     1. Gaussian measurement noise (SNR-parameterised)
%     2. Electrode contact impedance variation (multiplicative)
%     3. Systematic drift (random walk or linear)
%     4. Electrode bias (linear gradient)
%     5. Quantisation noise (ADC bit-depth)
%
%   Parameters:
%     dv     - (n_meas x 1) clean voltage difference vector
%     params - struct with fields:
%       .gaussian.enabled, .gaussian.snr_db
%       .contact_impedance.enabled, .contact_impedance.std_percent
%       .drift.enabled, .drift.rate_per_sample, .drift.max_magnitude
%       .electrode_bias.enabled, .electrode_bias.max_bias
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

    dv_noisy = dv;
    n_meas = length(dv);

    % --- 1. Gaussian measurement noise ---
    if params.gaussian.enabled
        snr_db = params.gaussian.snr_db;
        signal_power = norm(dv);
        if signal_power > 0
            noise = randn(n_meas, 1);
            scale = signal_power / norm(noise) * 10^(-snr_db / 20);
            dv_noisy = dv_noisy + scale * noise;
        end
    end

    % --- 2. Electrode contact impedance variation ---
    if params.contact_impedance.enabled
        std_frac = params.contact_impedance.std_percent / 100;
        % Multiplicative per-measurement factor (log-normal)
        impedance_factor = exp(std_frac * randn(n_meas, 1));
        dv_noisy = dv_noisy .* impedance_factor;
    end

    % --- 3. Systematic drift ---
    if params.drift.enabled
        rate = params.drift.rate_per_sample;
        max_mag = params.drift.max_magnitude;
        % Random walk drift (independent per call, no persistent state)
        drift_vec = cumsum(rate * randn(n_meas, 1));
        % Clip to maximum magnitude
        drift_vec = max(min(drift_vec, max_mag), -max_mag);
        dv_noisy = dv_noisy + drift_vec;
    end

    % --- 4. Electrode bias ---
    if params.electrode_bias.enabled
        max_bias = params.electrode_bias.max_bias;
        bias = linspace(-max_bias, max_bias, n_meas)';
        dv_noisy = dv_noisy + bias;
    end

    % --- 5. Quantisation noise ---
    if params.quantisation.enabled
        n_bits = params.quantisation.adc_bits;
        v_range = params.quantisation.voltage_range;
        lsb = v_range / (2^n_bits);  % Least significant bit
        % Uniform quantisation noise: [-LSB/2, +LSB/2]
        quant_noise = (rand(n_meas, 1) - 0.5) * lsb;
        dv_noisy = dv_noisy + quant_noise;
    end
end