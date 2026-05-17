%% Setup EIDORS for EIT Touch Classification Project
% Run this script once to configure EIDORS paths.
%
% Prerequisites:
%   1. Download EIDORS from: http://eidors3d.sourceforge.net/
%   2. Extract to: matlab/eidors/ (or update path below)
%   3. Ensure Netgen is installed for 3D mesh generation

function setup_eidors()
    project_root = fileparts(mfilename('fullpath'));

    % EIDORS path (adjust if installed elsewhere)
    eidors_dir = fullfile(project_root, 'eidors');

    if ~exist(eidors_dir, 'dir')
        error('setup_eidors:notFound', [...
            'EIDORS not found at: %s\n' ...
            'Download from: http://eidors3d.sourceforge.net/\n' ...
            'Extract to: %s'], eidors_dir, eidors_dir);
    end

    % Run EIDORS startup
    startup_file = fullfile(eidors_dir, 'startup.m');
    if exist(startup_file, 'file')
        run(startup_file);
        fprintf('EIDORS initialised successfully from: %s\n', eidors_dir);
    else
        error('setup_eidors:startupMissing', ...
            'EIDORS startup.m not found at: %s', startup_file);
    end

    % Add project paths
    addpath(fullfile(project_root, 'utils'));
    addpath(fullfile(project_root, 'noise_model'));
    addpath(fullfile(project_root, 'configs'));

    fprintf('Project paths added.\n');
end
