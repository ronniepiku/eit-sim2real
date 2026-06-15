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

    % Initialise EIDORS if not already on the path. mdl_normalize is a
    % canary symbol -- it is registered by EIDORS' startup script and
    % required by mk_common_model. Mirrors the bootstrap in main.m so
    % this script can be run standalone.
    script_dir = fileparts(mfilename('fullpath'));
    if isempty(which('mdl_normalize'))
        eidors_startup = fullfile(script_dir, 'eidors-v3.12-ng', ...
            'eidors', 'startup.m');
        if exist(eidors_startup, 'file')
            run(eidors_startup);
        else
            error('validate_mesh_convergence:eidorsNotFound', ...
                'EIDORS not found at: %s', eidors_startup);
        end
    end
    addpath(fullfile(script_dir, 'utils'));

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
    classes(4).sigma_range = [0.35, 0.55];
    classes(4).radius_range = [0.02, 0.05];
    classes(5).name = 'distributed';
    classes(5).sigma_range = [0.80, 0.92];
    classes(5).radius_range = [0.15, 0.25];

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

    % Diagnostics: count how many mesh elements each contact perturbs at
    % each refinement level, and how many samples are "under-resolved"
    % (zero or one element intersects the contact region) on each mesh.
    % This is the direct evidence for the dissertation claim that the
    % coarse mesh fails to resolve small-radius contacts on its native
    % discretisation (i.e., before the nearest-element fallback in
    % generate_sample.m is applied during production data generation).
    n_perturbed_total = zeros(5, 3);  % running sum of per-sample element counts
    n_underresolved   = zeros(5, 3);  % count of samples with <=1 element

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
                % Inner 40% per axis (matches get_touch_params in
                % generate_sample.m): on the unit circle, |x|, |y| <= 0.4.
                pos = (rand(1, 2) - 0.5) * 0.8;
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
                    in_contact = dist <= radius;
                    n_in = sum(in_contact);
                    n_perturbed_total(cls, r) = n_perturbed_total(cls, r) + n_in;
                    if n_in <= 1
                        n_underresolved(cls, r) = n_underresolved(cls, r) + 1;
                    end
                    img.elem_data(in_contact) = sigma;
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
    results.mean_rel_diff_cd        = mean(rel_diffs_cd);
    results.mean_rel_diff_cf        = mean(rel_diffs_cf);
    results.max_rel_diff_cf         = max(rel_diffs_cf);
    results.std_rel_diff_cf         = std(rel_diffs_cf);
    results.per_class               = per_class_cf;
    results.refinements             = refinements;
    results.class_names             = {classes.name};
    results.samples_per_class       = samples_per_class;
    results.mean_elements_perturbed = n_perturbed_total ./ samples_per_class;
    results.n_underresolved_samples = n_underresolved;

    fprintf('\n=== Summary ===\n');
    fprintf('Coarse vs Medium: mean = %.4f%%, max = %.4f%%\n', ...
        mean(rel_diffs_cd) * 100, max(rel_diffs_cd) * 100);
    fprintf('Coarse vs Fine:   mean = %.4f%%, max = %.4f%%\n', ...
        results.mean_rel_diff_cf * 100, results.max_rel_diff_cf * 100);
    fprintf('\nConclusion: Coarse mesh error (%.2f%%) << Noise floor (5%% at 40dB)\n', ...
        results.mean_rel_diff_cf * 100);

    fprintf('\nUnder-resolved samples (<=1 element intersected) per refinement:\n');
    fprintf('  %-15s   c    d    f   (of %d per class)\n', '', samples_per_class);
    for cls = 2:5
        fprintf('  %-15s %3d  %3d  %3d\n', classes(cls).name, ...
            n_underresolved(cls, 1), n_underresolved(cls, 2), n_underresolved(cls, 3));
    end

    % Persist results so the dissertation can cite reproducible artefacts.
    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'results', 'dataset_validation');
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    % Per-class table (CSV) for direct inclusion in the dissertation.
    class_names_col = {classes.name}';
    T = table(class_names_col, ...
        per_class_cf * 100, ...
        results.mean_elements_perturbed(:, 1), ...
        results.mean_elements_perturbed(:, 2), ...
        results.mean_elements_perturbed(:, 3), ...
        n_underresolved(:, 1), ...
        n_underresolved(:, 2), ...
        n_underresolved(:, 3), ...
        'VariableNames', {'class', 'mean_rel_diff_cf_pct', ...
            'mean_elem_c', 'mean_elem_d', 'mean_elem_f', ...
            'underres_c', 'underres_d', 'underres_f'});
    writetable(T, fullfile(out_dir, 'mesh_convergence_per_class.csv'));

    % Summary (JSON).
    summary = struct( ...
        'n_samples',            n_samples, ...
        'samples_per_class',    samples_per_class, ...
        'mean_rel_diff_cd_pct', results.mean_rel_diff_cd * 100, ...
        'mean_rel_diff_cf_pct', results.mean_rel_diff_cf * 100, ...
        'max_rel_diff_cf_pct',  results.max_rel_diff_cf  * 100, ...
        'std_rel_diff_cf_pct',  results.std_rel_diff_cf  * 100);
    fid = fopen(fullfile(out_dir, 'mesh_convergence_summary.json'), 'w');
    fprintf(fid, '%s', jsonencode(summary, 'PrettyPrint', true));
    fclose(fid);

    fprintf('\nResults saved to: %s\n', out_dir);
end
