function generate_mesh_refinement_testset(refinement, samples_per_class, seed)
%GENERATE_MESH_REFINEMENT_TESTSET Generate a test set at an alternative mesh refinement.
%
%   GENERATE_MESH_REFINEMENT_TESTSET() generates 100 samples per class at
%   refinement 'f' (fine) with seed 20260615.
%
%   GENERATE_MESH_REFINEMENT_TESTSET(refinement, samples_per_class, seed)
%   uses the supplied refinement level, sample count and seed.
%
%   Purpose:
%     Produces an out-of-mesh test set for the cross-mesh evaluation
%     experiment described in the dissertation's mesh-convergence
%     discussion (Section 5.2). The dataset is generated through the
%     EXACT same pipeline as the production training set
%     (create_mesh -> generate_sample -> add_noise), with only the
%     mesh refinement and seed differing. This isolates the effect of
%     mesh-induced clean-signal recalibration on classifier accuracy.
%
%   Inputs:
%     refinement        - EIDORS refinement level: 'c' (coarse, default
%                         production), 'd' (medium), or 'f' (fine).
%                         Default: 'f'.
%     samples_per_class - Samples generated per class (5 classes total).
%                         Default: 100 (-> 500 samples total).
%     seed              - RNG seed. Use a value distinct from the main
%                         dataset seed (42) so the test set is
%                         realisation-independent of the training set.
%                         Default: 20260615.
%
%   Output:
%     Writes data/eit_dataset_mesh_<refinement>.mat with the same key
%     layout as the production dataset:
%       dataset_X_clean   (n_samples x n_meas)
%       dataset_X_noisy   (n_samples x n_meas)
%       dataset_y         (n_samples x 1, 1-indexed labels)
%       dataset_metadata  (cell array)
%       config            (struct, including mesh refinement + seed)
%       noise_params      (struct, applied noise model)
%
%   The output file is consumed by the Python CLI:
%     uv run eit-sim2real experiments mesh-refinement

    arguments
        refinement (1,:) char {mustBeMember(refinement, {'c','d','f'})} = 'f'
        samples_per_class (1,1) double {mustBePositive, mustBeInteger} = 100
        seed (1,1) double {mustBeNonnegative} = 20260615
    end

    script_dir = fileparts(mfilename('fullpath'));

    %% Initialise EIDORS (canary symbol check, mirrors main.m)
    if isempty(which('mdl_normalize'))
        eidors_startup = fullfile(script_dir, 'eidors-v3.12-ng', ...
            'eidors', 'startup.m');
        if exist(eidors_startup, 'file')
            run(eidors_startup);
        else
            error('generate_mesh_refinement_testset:eidorsNotFound', ...
                'EIDORS not found at: %s', eidors_startup);
        end
    end
    addpath(fullfile(script_dir, 'utils'));
    addpath(fullfile(script_dir, 'noise_model'));

    %% Configuration (mirrors main.m, with refinement and seed swapped)
    config.samples_per_class = samples_per_class;
    config.n_classes         = 5;
    config.class_names       = {'none', 'light', 'firm', 'point', 'distributed'};
    config.output_dir        = fullfile(script_dir, '..', 'data');
    config.seed              = seed;
    config.mesh.geometry     = '2d_circle';
    config.mesh.n_elec       = 16;
    config.mesh.n_rings      = 2;
    config.mesh.refinement   = refinement;
    config.purpose           = 'mesh_refinement_evaluation_testset';

    %% Create mesh at the requested refinement
    fprintf('Creating mesh (refinement: %s, electrodes: %d)...\n', ...
        config.mesh.refinement, config.mesh.n_elec);
    [fmdl, vh] = create_mesh(config.mesh);
    n_meas = length(vh.meas);
    n_elem = size(fmdl.elems, 1);
    fprintf('  Mesh created: %d elements, %d measurements\n', n_elem, n_meas);

    %% Load noise parameters (same model used for production data)
    noise_params = load_noise_params();
    fprintf('Noise model loaded (gaussian=%d, impedance=%d, bias=%d, quant=%d)\n', ...
        noise_params.gaussian.enabled, ...
        noise_params.contact_impedance.enabled, ...
        noise_params.electrode_bias.enabled, ...
        noise_params.quantisation.enabled);

    %% Generate dataset
    rng(config.seed);
    total_samples = config.samples_per_class * config.n_classes;
    fprintf('Generating %d test samples (%d per class) at refinement %s...\n', ...
        total_samples, config.samples_per_class, refinement);

    dataset_X_noisy  = zeros(total_samples, n_meas);
    dataset_X_clean  = zeros(total_samples, n_meas);
    dataset_y        = zeros(total_samples, 1);
    dataset_metadata = cell(total_samples, 1);

    sample_idx = 0;
    for c = 1:config.n_classes
        class_name = config.class_names{c};
        fprintf('  Class %d/%d (%s): ', c, config.n_classes, class_name);

        for s = 1:config.samples_per_class
            sample_idx = sample_idx + 1;

            [dv_noisy, dv_clean, class_id, meta] = generate_sample( ...
                fmdl, vh, noise_params, class_name);

            dataset_X_noisy(sample_idx, :) = dv_noisy';
            dataset_X_clean(sample_idx, :) = dv_clean';
            dataset_y(sample_idx)          = class_id;
            dataset_metadata{sample_idx}   = meta;

            if mod(s, max(1, floor(config.samples_per_class / 10))) == 0
                fprintf('.');
            end
        end
        fprintf(' done\n');
    end

    %% Save dataset
    if ~exist(config.output_dir, 'dir')
        mkdir(config.output_dir);
    end
    out_name  = sprintf('eit_dataset_mesh_%s.mat', refinement);
    out_path  = fullfile(config.output_dir, out_name);
    fprintf('Saving dataset to: %s\n', out_path);
    save(out_path, 'dataset_X_noisy', 'dataset_X_clean', 'dataset_y', ...
        'dataset_metadata', 'config', 'noise_params', '-v7');

    %% Summary
    fprintf('\n=== Mesh-Refinement Test Set Summary ===\n');
    fprintf('Refinement level: %s (%d elements)\n', refinement, n_elem);
    fprintf('Total samples:    %d\n', total_samples);
    fprintf('Per-class count:  %d\n', config.samples_per_class);
    fprintf('Seed:             %d\n', config.seed);
    fprintf('Output:           %s\n', out_path);
    fprintf('========================================\n');
end
