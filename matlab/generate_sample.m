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
    dv_clean = vi.meas - vh.meas;

    % Apply noise model
    dv_noisy = add_noise(dv_clean, noise_params);

    % Encode label
    [class_id, ~, metadata] = encode_label(touch_type, touch_params);
end


function params = get_touch_params(touch_type, fmdl)
%GET_TOUCH_PARAMS Generate physical parameters for a given touch class.
%
%   Parameters are sampled from defined ranges to create variability
%   within each class while maintaining class-defining characteristics.

    % Determine valid position range from mesh bounds
    nodes = fmdl.nodes;
    x_range = [min(nodes(:,1)), max(nodes(:,1))] * 0.7;  % Stay within 70% of boundary
    y_range = [min(nodes(:,2)), max(nodes(:,2))] * 0.7;

    % Random position (shared across classes that have contact)
    x = x_range(1) + rand() * (x_range(2) - x_range(1));
    y = y_range(1) + rand() * (y_range(2) - y_range(1));

    switch touch_type
        case 'none'
            % No contact: baseline measurement
            params = struct('radius', 0, 'conductivity', 1.0, 'x', 0, 'y', 0);

        case 'light'
            % Light touch: small conductivity change, medium area
            radius = 0.08 + 0.04 * rand();         % [0.08, 0.12]
            conductivity = 1.1 + 0.15 * rand();    % [1.1, 1.25]
            params = struct('radius', radius, 'conductivity', conductivity, ...
                'x', x, 'y', y);

        case 'firm'
            % Firm press: large conductivity change, medium area
            radius = 0.08 + 0.04 * rand();         % [0.08, 0.12]
            conductivity = 1.8 + 0.4 * rand();     % [1.8, 2.2]
            params = struct('radius', radius, 'conductivity', conductivity, ...
                'x', x, 'y', y);

        case 'point'
            % Point contact: medium force, very small area
            radius = 0.02 + 0.02 * rand();         % [0.02, 0.04]
            conductivity = 1.4 + 0.3 * rand();     % [1.4, 1.7]
            params = struct('radius', radius, 'conductivity', conductivity, ...
                'x', x, 'y', y);

        case 'distributed'
            % Distributed contact: medium force, large area
            radius = 0.15 + 0.08 * rand();         % [0.15, 0.23]
            conductivity = 1.3 + 0.2 * rand();     % [1.3, 1.5]
            params = struct('radius', radius, 'conductivity', conductivity, ...
                'x', x, 'y', y);
    end
end