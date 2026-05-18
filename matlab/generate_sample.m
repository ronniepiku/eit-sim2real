function [dv_noisy, dv_clean, class_id, metadata] = generate_sample(fmdl, vh, noise_params, touch_type)
%GENERATE_SAMPLE Generate a single EIT measurement sample with noise.
%
%   [dv_noisy, dv_clean, class_id, metadata] = GENERATE_SAMPLE(fmdl, vh, noise_params)
%   generates a random touch sample from a randomly selected class.
%
%   [dv_noisy, dv_clean, class_id, metadata] = GENERATE_SAMPLE(fmdl, vh, noise_params, touch_type)
%   generates a sample from the specified touch class.
%
%   Parameters:
%     fmdl         - EIDORS forward model
%     vh           - Baseline voltage measurements (homogeneous)
%     noise_params - Noise parameters struct (from load_noise_params)
%     touch_type   - (optional) Force specific class: 'none', 'light',
%                    'firm', 'point', 'distributed'
%
%   Returns:
%     dv_noisy  - (n_meas x 1) noisy voltage difference vector
%     dv_clean  - (n_meas x 1) clean voltage difference (for reference)
%     class_id  - Integer class label (1-5)
%     metadata  - Struct with all generation parameters

    arguments
        fmdl struct
        vh struct
        noise_params struct
        touch_type (1,:) char = ''
    end

    % Select touch class if not specified
    if isempty(touch_type)
        classes = {'none', 'light', 'firm', 'point', 'distributed'};
        touch_type = classes{randi(5)};
    end

    % Get touch parameters for this class
    touch_params = get_touch_params(touch_type, fmdl);

    % Create image with conductivity change
    img = mk_image(fmdl, 1);  % Background conductivity

    if ~strcmp(touch_type, 'none')
        % Get element centres
        elem_centres = get_element_centers(fmdl);

        % Define contact region (circular)
        dist = sqrt((elem_centres(:,1) - touch_params.x).^2 + ...
                    (elem_centres(:,2) - touch_params.y).^2);
        contact_mask = dist < touch_params.radius;

        % Apply conductivity change to contact region
        img.elem_data(contact_mask) = touch_params.conductivity;
    end

    % Forward solve
    vi = fwd_solve(img);

    % Voltage difference from baseline
    dv_clean = double(vi.meas - vh.meas);

    % Apply noise model
    dv_noisy = add_noise(dv_clean, noise_params);

    % Encode label
    [class_id, ~, metadata] = encode_label(touch_type, touch_params);
end


function params = get_touch_params(touch_type, fmdl)
%GET_TOUCH_PARAMS Generate physical parameters for a given touch class.
%
%   Physics-informed parameterisation for an ionic hydrogel EIT e-skin.
%
%   Material model:
%     Ionic hydrogel exhibits POSITIVE piezoresistive response — local
%     resistance increases (conductivity DECREASES) under mechanical load.
%     This arises from compression disrupting ion-transport pathways in
%     the hydrated polymer matrix.
%
%   Evidence basis (see defensibility table in methodology chapter):
%     - Sign (sigma < sigma_0): [19] Lee et al. — ionic hydrogel layer
%       shows "positive piezoresistive response (resistance up with strain)"
%     - Force envelope 0.01-10 N: [55] prosthetic tactile design constraint
%     - Demonstrated hydrogel EIT range 0.5-2 N: [20] FISTA paper (50-200 g)
%     - Detection threshold >= 2% change: [38] Boone & Holder
%     - Contact radii from Hertzian model with E* ~ 150 kPa: [18] Kim et al.
%
%   Calibration rules:
%     - Magnitudes are physics-informed priors, not directly measured constants
%     - Class separability validated via forward-solve signal norms
%     - Conductivity change proportional to local pressure: Δσ/σ₀ ∝ F/(πr²)
%     - Ranges ensure: point > firm > distributed ≈ light in |Δσ| per element
%     - Position constrained to 40% of boundary to limit sensitivity-gradient
%       induced variance that would otherwise dominate class differences

    % Determine valid position range from mesh bounds
    % Constrain to 40% of boundary radius to reduce the EIT sensitivity
    % gradient's impact on within-class variance (sensitivity is highest
    % near electrodes and lowest in centre; 70% allowed too much spread)
    nodes = fmdl.nodes;
    x_range = [min(nodes(:,1)), max(nodes(:,1))] * 0.4;
    y_range = [min(nodes(:,2)), max(nodes(:,2))] * 0.4;

    % Random position (shared across classes that have contact)
    x = x_range(1) + rand() * (x_range(2) - x_range(1));
    y = y_range(1) + rand() * (y_range(2) - y_range(1));

    switch touch_type
        case 'none'
            % No contact: baseline measurement (homogeneous hydrogel)
            params = struct('radius', 0, 'conductivity', 1.0, 'x', 0, 'y', 0);

        case 'light'
            % Light touch (~0.1–0.5 N): mild conductivity decrease, medium area
            % Local pressure P ≈ F/(πr²) ≈ 0.3/(π×0.08²) ≈ 15 kPa
            % Evidence: [3] hydrogel recovery under 2 kPa; [55] low-end force
            % Conductivity drop 5–15% above detection threshold ([38]: 2%)
            radius = 0.06 + 0.04 * rand();         % [0.06, 0.10]
            conductivity = 0.85 + 0.10 * rand();   % [0.85, 0.95]
            params = struct('radius', radius, 'conductivity', conductivity, ...
                'x', x, 'y', y);

        case 'firm'
            % Firm press (~1.0–3.0 N): strong conductivity decrease, medium area
            % Local pressure P ≈ 2.0/(π×0.10²) ≈ 64 kPa
            % Evidence: [20] 50–200 g on hydrogel; [10] tested to 2.5 N
            % Conductivity drop 25–45%, within linear EIT regime ([17])
            radius = 0.08 + 0.04 * rand();         % [0.08, 0.12]
            conductivity = 0.55 + 0.20 * rand();   % [0.55, 0.75]
            params = struct('radius', radius, 'conductivity', conductivity, ...
                'x', x, 'y', y);

        case 'point'
            % Point contact (~0.3–1.5 N): moderate force, very small area
            % Local pressure P ≈ 0.8/(π×0.03²) ≈ 280 kPa (very high)
            % Evidence: [64] poke modality; high local pressure on small patch
            % Physics: F/A very large → strongest local Δσ of all classes
            % Conductivity drop 45–65% justified by extreme local pressure
            radius = 0.02 + 0.03 * rand();         % [0.02, 0.05]
            conductivity = 0.35 + 0.20 * rand();   % [0.35, 0.55]
            params = struct('radius', radius, 'conductivity', conductivity, ...
                'x', x, 'y', y);

        case 'distributed'
            % Distributed contact (~1.0–4.0 N): moderate force, large area
            % Local pressure P ≈ 2.0/(π×0.20²) ≈ 16 kPa (low per element)
            % Evidence: [14] distributed pressure; [55] grasping forces
            % Physics: F spread over large area → small per-element Δσ but
            % distinct spatial signature (broad, shallow perturbation)
            % Conductivity drop 8–20% per element over wide region
            radius = 0.15 + 0.10 * rand();         % [0.15, 0.25]
            conductivity = 0.80 + 0.12 * rand();   % [0.80, 0.92]
            params = struct('radius', radius, 'conductivity', conductivity, ...
                'x', x, 'y', y);
    end
end