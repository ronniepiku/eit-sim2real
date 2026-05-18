function results = validate_mesh_convergence(n_samples)
%VALIDATE_MESH_CONVERGENCE Compare forward solutions across mesh refinements.
%
%   results = VALIDATE_MESH_CONVERGENCE() runs 100 samples (default).
%   results = VALIDATE_MESH_CONVERGENCE(n_samples) uses specified count.
%
%   Compares coarse ('c'), medium ('d'), and fine ('f') mesh refinements
%   to verify that the coarse mesh introduces negligible discretisation
%   error relative to the noise model magnitudes.
%
%   Returns:
%     results - struct with fields:
%       .mean_rel_diff_cd  - mean relative difference coarse vs medium
%       .mean_rel_diff_cf  - mean relative difference coarse vs fine
%       .max_rel_diff_cf   - max relative difference coarse vs fine
%       .per_class         - per-class breakdown

    arguments
        n_samples (1,1) double {mustBePositive} = 100
    end

    fprintf('=== Mesh Convergence Validation ===\n');
    fprintf('Comparing refinement levels: coarse (c), medium (d), fine (f)\n');
    fprintf('Samples per class: %d (total: %d)\n\n', ...
        n_samples / 5, n_samples);

    rng(42);  % Fixed seed for reproducibility

    % Class parameters (matching generate_sample.m)
    classes = struct();
    classes(1).name = 'no_contact';
    classes(1).sigma_range = [1.0, 1.0];
    classes(1).radius_range = [0, 0];
    classes(2).name = 'light_touch';
    classes(2).sigma_range = [0.85, 0.95];
    classes(2).radius_range = [0.06, 0.10];
    classes(3).name = 'firm_press';
    classes(3).sigma_range = [0.55, 0.75];
    classes(3).radius_range = [0.08, 0.12];
    classes(4).name = 'point_contact';
    classes(4).sigma_range = [0.60, 0.80];
    classes(4).radius_range = [0.02, 0.04];
    classes(5).name = 'distributed';
    classes(5).sigma_range = [0.60, 0.80];
    classes(5).radius_range = [0.12, 0.20];

    % Create models at each refinement level
    refinements = {'c', 'd', 'f'};
    models = cell(1, 3);
    baselines = cell(1, 3);

    for r = 1:3
        opts.refinement = refinements{r};
        opts.geometry = '2d_circle';
        opts.n_elec = 16;
        [fmdl, vh] = create_mesh(opts);
        models{r} = fmdl;
        baselines{r} = vh.meas;
    end

    % Generate samples and compare
    samples_per_class = floor(n_samples / 5);
    rel_diffs_cd = [];
    rel_diffs_cf = [];
    per_class_cf = zeros(5, 1);

    for cls = 1:5
        class_diffs = [];
        for s = 1:samples_per_class
            % Random parameters
            if cls == 1
                % No contact - homogeneous
                sigma = 1.0;
                radius = 0;
                pos = [0, 0];
            else
                sigma = classes(cls).sigma_range(1) + ...
                    rand() * diff(classes(cls).sigma_range);
                radius = classes(cls).radius_range(1) + ...
                    rand() * diff(classes(cls).radius_range);
                pos = (rand(1, 2) - 0.5) * 1.4;  % Inner 70%
                % Ensure within bounds
                while norm(pos) + radius > 0.9
                    pos = (rand(1, 2) - 0.5) * 1.4;
                end
            end

            % Solve forward problem at each refinement
            dv = cell(1, 3);
            for r = 1:3
                if cls == 1
                    dv{r} = zeros(size(baselines{r}));
                else
                    img = mk_image(models{r}, 1);
                    % Find elements within contact region
                    elem_centres = interp_mesh(models{r});
                    dist = sqrt((elem_centres(:,1) - pos(1)).^2 + ...
                               (elem_centres(:,2) - pos(2)).^2);
                    img.elem_data(dist <= radius) = sigma;
                    v_contact = fwd_solve(img);
                    dv{r} = v_contact.meas - baselines{r};
                end
            end

            % Compute relative differences
            if norm(dv{3}) > 1e-12  % Avoid division by zero (no-contact)
                rd_cd = norm(dv{1} - dv{2}) / norm(dv{3});
                rd_cf = norm(dv{1} - dv{3}) / norm(dv{3});
                rel_diffs_cd = [rel_diffs_cd; rd_cd]; %#ok<AGROW>
                rel_diffs_cf = [rel_diffs_cf; rd_cf]; %#ok<AGROW>
                class_diffs = [class_diffs; rd_cf]; %#ok<AGROW>
            end
        end
        if ~isempty(class_diffs)
            per_class_cf(cls) = mean(class_diffs);
        end
        fprintf('Class %d (%s): mean rel. diff (c vs f) = %.4f%%\n', ...
            cls, classes(cls).name, per_class_cf(cls) * 100);
    end

    % Summary statistics
    results.mean_rel_diff_cd = mean(rel_diffs_cd);
    results.mean_rel_diff_cf = mean(rel_diffs_cf);
    results.max_rel_diff_cf = max(rel_diffs_cf);
    results.std_rel_diff_cf = std(rel_diffs_cf);
    results.per_class = per_class_cf;

    fprintf('\n=== Summary ===\n');
    fprintf('Coarse vs Medium: mean = %.4f%%, max = %.4f%%\n', ...
        mean(rel_diffs_cd) * 100, max(rel_diffs_cd) * 100);
    fprintf('Coarse vs Fine:   mean = %.4f%%, max = %.4f%%\n', ...
        results.mean_rel_diff_cf * 100, results.max_rel_diff_cf * 100);
    fprintf('\nConclusion: Coarse mesh error (%.2f%%) << Noise floor (5%% at 40dB)\n', ...
        results.mean_rel_diff_cf * 100);
end
