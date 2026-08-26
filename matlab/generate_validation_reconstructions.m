function results = generate_validation_reconstructions(data_path, output_dir, n_mean_samples)
%GENERATE_VALIDATION_RECONSTRUCTIONS Create EIDORS inverse-reconstruction figures.
%
%   results = GENERATE_VALIDATION_RECONSTRUCTIONS()
%   results = GENERATE_VALIDATION_RECONSTRUCTIONS(data_path)
%   results = GENERATE_VALIDATION_RECONSTRUCTIONS(data_path, output_dir)
%   results = GENERATE_VALIDATION_RECONSTRUCTIONS(data_path, output_dir, n_mean_samples)
%
%   Generates dissertation-ready figures showing:
%     - one random reconstructed image per class
%     - one mean reconstructed image per class
%   for both clean and noisy dataset variants.
%
%   Output figures are saved as PNG and PDF in:
%     results/dataset_validation/reconstructions/
%
%   Notes:
%     - This script requires EIDORS to be initialised (run setup_eidors).
%     - Reconstructions use a coarser inverse model to avoid inverse crime.
%     - Mean images are computed as the average of multiple reconstructed
%       images from a stratified random subset of each class.

    arguments
        data_path char = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'eit_dataset.mat')
        output_dir char = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'dataset_validation', 'reconstructions')
        n_mean_samples (1,1) double {mustBePositive} = 25
    end

    if exist('eidors_obj', 'file') ~= 2
        setup_eidors();
    end

    if exist(data_path, 'file') ~= 2
        error('generate_validation_reconstructions:datasetNotFound', ...
            'Dataset not found: %s', data_path);
    end

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    rng(42);  % Fixed seed for reproducibility, consistent with the rest of the pipeline

    data = load(data_path, 'dataset_X_clean', 'dataset_X_noisy', 'dataset_y', 'config');
    class_names = {'No contact', 'Light touch', 'Firm press', 'Point contact', 'Distributed'};
    class_ids = 1:5;

    variants = struct('name', {'clean', 'noisy'}, ...
                      'field', {'dataset_X_clean', 'dataset_X_noisy'});

    results = struct();

    for v = 1:numel(variants)
        variant_name = variants(v).name;
        X = data.(variants(v).field);
        y = data.dataset_y;

        fprintf('Generating reconstructions for %s data...\n', variant_name);

        % Create forward and inverse models
        [fmdl, vh] = create_mesh(data.config.mesh);
        inv2d = eidors_obj('inv_model', 'EIT inverse');
        inv2d.reconst_type = 'difference';
        inv2d.jacobian_bkgnd.value = 1;
        inv2d.solve = @inv_solve_diff_GN_one_step;
        inv2d.hyperparameter.value = 0.003;
        inv2d.RtR_prior = @prior_laplace;

        % Avoid inverse crime by using a coarser inverse forward model
        inv_mdl = mk_common_model('b2c', 16);
        inv2d.fwd_model = inv_mdl.fwd_model;

        % Gather reconstructions
        random_imgs = cell(numel(class_ids), 1);
        mean_imgs = cell(numel(class_ids), 1);
        random_samples = zeros(numel(class_ids), 1);

        for c = 1:numel(class_ids)
            class_id = class_ids(c);
            class_idx = find(y == class_id);
            if isempty(class_idx)
                error('generate_validation_reconstructions:noSamples', ...
                    'No samples found for class %d (%s)', class_id, class_names{c});
            end

            % Random representative sample
            random_pick = class_idx(randi(numel(class_idx)));
            random_samples(c) = random_pick;
            random_imgs{c} = reconstruct_sample(inv2d, vh, X(random_pick, :));

            % Mean reconstruction over a stratified subset of samples
            n_pick = min(n_mean_samples, numel(class_idx));
            perm = randperm(numel(class_idx), n_pick);
            img_sum = [];
            template_img = [];
            for i = 1:n_pick
                img = reconstruct_sample(inv2d, vh, X(class_idx(perm(i)), :));
                if isempty(img_sum)
                    img_sum = zeros(size(img.elem_data));
                    template_img = img;
                end
                img_sum = img_sum + double(img.elem_data(:));
            end
            mean_img = template_img;
            mean_img.elem_data = img_sum / n_pick;
            mean_imgs{c} = mean_img;
        end

        % Save combined figure set
        random_prefix = fullfile(output_dir, sprintf('random_reconstructed_class_images_%s', variant_name));
        mean_prefix = fullfile(output_dir, sprintf('mean_reconstructed_class_images_%s', variant_name));

        save_reconstruction_grid(random_imgs, class_names, sprintf('Random Reconstructed Class Images (%s)', variant_name), random_prefix);
        save_reconstruction_grid(mean_imgs, class_names, sprintf('Mean Reconstructed Class Images (%s)', variant_name), mean_prefix);

        results.(variant_name).random_samples = random_samples;
        results.(variant_name).random_images = random_imgs;
        results.(variant_name).mean_images = mean_imgs;
        results.(variant_name).random_figure = random_prefix;
        results.(variant_name).mean_figure = mean_prefix;
    end

    save(fullfile(output_dir, 'validation_reconstructions.mat'), 'results', '-v7.3');
    fprintf('Saved validation reconstructions to %s\n', output_dir);
end


function img = reconstruct_sample(inv2d, vh, dv_row)
%RECONSTRUCT_SAMPLE Run a single EIDORS difference reconstruction.
    vi = vh;
    vi.meas = vh.meas + dv_row(:);
    img = inv_solve(inv2d, vh, vi);
end


function save_reconstruction_grid(images, class_names, fig_title, file_prefix)
%SAVE_RECONSTRUCTION_GRID Render and save a 2x3 reconstruction grid.
    fig = figure('Color', 'w', 'Visible', 'off');
    colormap('jet');

    % Determine a common colour scale for comparability.
    all_vals = [];
    for i = 1:numel(images)
        all_vals = [all_vals; double(images{i}.elem_data(:))]; %#ok<AGROW>
    end
    clim = [min(all_vals), max(all_vals)];
    if clim(1) == clim(2)
        clim = clim + [-1, 1] * 1e-6;
    end

    for i = 1:numel(images)
        subplot(2, 3, i);
        show_slices(images{i});
        axis image off;
        title(class_names{i}, 'FontWeight', 'bold', 'Color', 'k', 'FontSize', 11);
        caxis(clim);
    end

    sgtitle(fig_title, 'FontWeight', 'bold', 'Color', 'k', 'FontSize', 13);
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, [file_prefix '.png'], '-dpng', '-r300');
    print(fig, [file_prefix '.pdf'], '-dpdf', '-painters');
    close(fig);
end
