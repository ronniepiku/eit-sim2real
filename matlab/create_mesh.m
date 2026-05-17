function [fmdl, vh] = create_mesh(opts)
%CREATE_MESH Create EIT forward model mesh and compute baseline voltages.
%
%   [fmdl, vh] = CREATE_MESH() creates a 2D circular model (default).
%   [fmdl, vh] = CREATE_MESH(opts) creates a model with specified options.
%
%   Options (struct fields):
%     .geometry   - '2d_circle' (default) or '3d_cylinder'
%     .n_elec     - Number of electrodes (default: 16)
%     .n_rings    - Number of electrode rings for 3D (default: 2)
%     .refinement - Mesh refinement level (default: 'c' for coarse)
%
%   Returns:
%     fmdl - EIDORS forward model struct
%     vh   - Baseline (homogeneous) voltage measurements
%
%   Examples:
%     [fmdl, vh] = create_mesh();
%     [fmdl, vh] = create_mesh(struct('geometry', '3d_cylinder', 'n_elec', 16));

    arguments
        opts struct = struct()
    end

    % Default options
    if ~isfield(opts, 'geometry'),   opts.geometry = '2d_circle'; end
    if ~isfield(opts, 'n_elec'),     opts.n_elec = 16; end
    if ~isfield(opts, 'n_rings'),    opts.n_rings = 2; end
    if ~isfield(opts, 'refinement'), opts.refinement = 'c'; end

    switch opts.geometry
        case '2d_circle'
            fmdl = create_2d_circle(opts);
        case '3d_cylinder'
            fmdl = create_3d_cylinder(opts);
        otherwise
            error('create_mesh:invalidGeometry', ...
                'Unknown geometry: %s. Use ''2d_circle'' or ''3d_cylinder''.', ...
                opts.geometry);
    end

    % Compute baseline (homogeneous) voltages
    img_bg = mk_image(fmdl, 1);  % Background conductivity = 1 S/m
    vh = fwd_solve(img_bg);
end


function fmdl = create_2d_circle(opts)
%CREATE_2D_CIRCLE Create a 2D circular cross-section model.

    % Model string: refinement + '2C' + number format
    model_str = sprintf('%s2C', opts.refinement);
    imdl = mk_common_model(model_str, opts.n_elec);
    fmdl = imdl.fwd_model;
end


function fmdl = create_3d_cylinder(opts)
%CREATE_3D_CYLINDER Create a 3D cylindrical model representing an arm.
%
%   Uses EIDORS ng_mk_cyl_models for a cylindrical mesh with electrode
%   rings, suitable for modelling a prosthetic arm cross-section.

    % Cylinder parameters
    height = 0.3;           % Cylinder height (m) - approximate arm segment
    radius = 0.05;          % Cylinder radius (m) - approximate forearm
    max_elem_size = 0.01;   % Maximum element size for mesh quality

    % Electrode parameters
    elecs_per_ring = opts.n_elec / opts.n_rings;
    elec_positions = [];
    for ring = 1:opts.n_rings
        z_pos = height * ring / (opts.n_rings + 1);
        for e = 1:elecs_per_ring
            angle = 2 * pi * (e - 1) / elecs_per_ring;
            elec_positions = [elec_positions; angle, z_pos]; %#ok<AGROW>
        end
    end

    % Create cylindrical mesh with electrodes
    elec_shape = [0.005, 0, 0.005];  % Electrode: 5mm diameter, point, 5mm height
    fmdl = ng_mk_cyl_models([height, radius, max_elem_size], ...
        elec_positions, elec_shape);

    % Set stimulation pattern: adjacent drive, adjacent measurement
    [stim, meas_sel] = mk_stim_patterns(opts.n_elec, 1, ...
        [0, 1], [0, 1], {'no_meas_current'}, 1);
    fmdl.stimulation = stim;
    fmdl.meas_select = meas_sel;
end