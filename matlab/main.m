%% EIT Touch Classification - Dataset Generation
% Generates a simulated EIT dataset with physically-motivated noise for
% training touch classifiers.
%
% Dissertation: "Towards Simulation-to-Reality Transfer in EIT Tactile
% Sensing: A Noise-Augmented Deep Learning Approach"
%
% This script generates both clean and noisy voltage difference vectors
% for 5 touch classes, with balanced class sampling.

clear; clc; close all;

%% Configuration
% --- Dataset parameters ---
config.samples_per_class = 1000;    % Samples per class (total = 5 * this)
config.n_classes = 5;
config.class_names = {'none', 'light', 'firm', 'point', 'distributed'};
config.output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
config.seed = 42;                    % For reproducibility

% --- Mesh options ---
config.mesh.geometry = '2d_circle';  % '2d_circle' or '3d_cylinder'
config.mesh.n_elec = 16;
config.mesh.n_rings = 2;
config.mesh.refinement = 'c';

%% Initialise
% Set random seed for reproducibility
rng(config.seed);

% Add paths
addpath(fullfile(fileparts(mfilename('fullpath')), 'utils'));
addpath(fullfile(fileparts(mfilename('fullpath')), 'noise_model'));

% Initialise EIDORS (adjust path as needed)
eidors_path = fullfile(fileparts(mfilename('fullpath')), 'eidors', 'startup.m');
if exist(eidors_path, 'file')
    run(eidors_path);
else
    error('main:eidorsNotFound', ...
        'EIDORS not found at: %s\nDownload from http://eidors3d.sourceforge.net/', ...
        eidors_path);
end

%% Create Mesh
fprintf('Creating mesh (geometry: %s, electrodes: %d)...\n', ...
    config.mesh.geometry, config.mesh.n_elec);
[fmdl, vh] = create_mesh(config.mesh);
n_meas = length(vh.meas);
fprintf('  Mesh created: %d elements, %d measurements\n', ...
    size(fmdl.elems, 1), n_meas);

%% Load Noise Parameters
noise_params = load_noise_params();
fprintf('Noise model loaded (components: gaussian=%d, impedance=%d, drift=%d, bias=%d, quant=%d)\n', ...
    noise_params.gaussian.enabled, ...
    noise_params.contact_impedance.enabled, ...
    noise_params.drift.enabled, ...
    noise_params.electrode_bias.enabled, ...
    noise_params.quantisation.enabled);

%% Generate Dataset
total_samples = config.samples_per_class * config.n_classes;
fprintf('Generating %d samples (%d per class)...\n', total_samples, config.samples_per_class);

% Pre-allocate arrays
dataset_X_noisy = zeros(total_samples, n_meas);
dataset_X_clean = zeros(total_samples, n_meas);
dataset_y = zeros(total_samples, 1);
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
        dataset_y(sample_idx) = class_id;
        dataset_metadata{sample_idx} = meta;

        if mod(s, 200) == 0
            fprintf('.');
        end
    end
    fprintf(' done\n');
end

%% Create Output Directory
if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
end

%% Save Dataset
output_file = fullfile(config.output_dir, 'eit_dataset.mat');
fprintf('Saving dataset to: %s\n', output_file);
save(output_file, 'dataset_X_noisy', 'dataset_X_clean', 'dataset_y', ...
    'dataset_metadata', 'config', 'noise_params', '-v7.3');

%% Save as NumPy-compatible format for Python pipeline
% Export as separate files for easy loading in Python
output_npz = fullfile(config.output_dir, 'eit_dataset_numpy.mat');
save(output_npz, 'dataset_X_noisy', 'dataset_X_clean', 'dataset_y', '-v7');

%% Summary Statistics
fprintf('\n=== Dataset Summary ===\n');
fprintf('Total samples:    %d\n', total_samples);
fprintf('Feature dim:      %d\n', n_meas);
fprintf('Classes:          %d\n', config.n_classes);
fprintf('Geometry:         %s\n', config.mesh.geometry);
fprintf('Noise enabled:    %s\n', mat2str(noise_params.gaussian.enabled));
for c = 1:config.n_classes
    count = sum(dataset_y == c);
    fprintf('  Class %d (%s): %d samples\n', c, config.class_names{c}, count);
end
fprintf('Output:           %s\n', output_file);
fprintf('========================\n');
