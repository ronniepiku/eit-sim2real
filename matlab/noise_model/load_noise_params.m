function params = load_noise_params(config_path)
%LOAD_NOISE_PARAMS Load noise parameters from YAML configuration file.
%
%   params = LOAD_NOISE_PARAMS(config_path) reads the YAML config and returns
%   a struct compatible with add_noise().
%
%   If no config_path is provided, uses the default config location.

    arguments
        config_path (1,1) string = fullfile(fileparts(mfilename('fullpath')), ...
            '..', 'configs', 'noise_params.yaml')
    end

    % Read YAML file (MATLAB R2019b+ has yaml support, fallback to manual parse)
    if exist('yaml.loadFile', 'file')
        raw = yaml.loadFile(char(config_path));
    else
        raw = parse_yaml_simple(config_path);
    end

    % Convert to struct with proper types
    params = struct();

    % Gaussian
    params.gaussian.enabled = raw.gaussian.enabled;
    params.gaussian.snr_db = raw.gaussian.snr_db;
    params.gaussian.noise_floor = raw.gaussian.noise_floor;

    % Contact impedance
    params.contact_impedance.enabled = raw.contact_impedance.enabled;
    params.contact_impedance.std_percent = raw.contact_impedance.std_percent;
    params.contact_impedance.n_electrodes = raw.contact_impedance.n_electrodes;

    % Electrode bias
    params.electrode_bias.enabled = raw.electrode_bias.enabled;
    params.electrode_bias.max_bias = raw.electrode_bias.max_bias;
    params.electrode_bias.n_electrodes = raw.electrode_bias.n_electrodes;

    % Quantisation
    params.quantisation.enabled = raw.quantisation.enabled;
    params.quantisation.adc_bits = raw.quantisation.adc_bits;
    params.quantisation.voltage_range = raw.quantisation.voltage_range;
end


function data = parse_yaml_simple(filepath)
%PARSE_YAML_SIMPLE Minimal YAML parser for flat/nested key-value configs.
%   Handles the noise_params.yaml structure specifically.

    fid = fopen(filepath, 'r');
    if fid == -1
        error('load_noise_params:fileNotFound', ...
            'Cannot open config file: %s', filepath);
    end
    cleanup = onCleanup(@() fclose(fid));

    data = struct();
    current_section = '';

    while ~feof(fid)
        line = fgetl(fid);
        if isempty(line) || startsWith(strtrim(line), '#')
            continue;
        end

        % Detect indentation level
        stripped = strtrim(line);
        indent = find(line ~= ' ', 1) - 1;

        if indent == 0 && contains(stripped, ':')
            % Top-level key
            parts = split(stripped, ':');
            key = strtrim(parts{1});
            val = strtrim(strjoin(parts(2:end), ':'));
            if isempty(val)
                current_section = key;
                if ~isfield(data, current_section)
                    data.(current_section) = struct();
                end
            else
                data.(key) = parse_value(val);
            end
        elseif indent > 0 && ~isempty(current_section) && contains(stripped, ':')
            % Nested key
            parts = split(stripped, ':');
            key = strtrim(parts{1});
            val = strtrim(strjoin(parts(2:end), ':'));
            if ~isempty(val)
                data.(current_section).(key) = parse_value(val);
            end
        end
    end
end


function val = parse_value(str)
%PARSE_VALUE Convert a YAML value string to appropriate MATLAB type.

    % Remove inline comments
    comment_idx = strfind(str, '#');
    if ~isempty(comment_idx)
        str = strtrim(str(1:comment_idx(1)-1));
    end

    % Boolean
    if strcmpi(str, 'true')
        val = true; return;
    elseif strcmpi(str, 'false')
        val = false; return;
    end

    % Array [a, b, c]
    if startsWith(str, '[') && endsWith(str, ']')
        inner = str(2:end-1);
        parts = split(inner, ',');
        val = cellfun(@(x) str2double(strtrim(x)), parts);
        return;
    end

    % Number
    num = str2double(str);
    if ~isnan(num)
        val = num; return;
    end

    % String (strip quotes if present)
    str = strrep(str, '''', '');
    str = strrep(str, '"', '');
    val = str;
end
