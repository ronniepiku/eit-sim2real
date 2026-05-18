function [class_id, class_name, metadata] = encode_label(touch_type, params)
%ENCODE_LABEL Encode touch parameters into a class label.
%
%   [class_id, class_name, metadata] = ENCODE_LABEL(touch_type, params)
%
%   Touch Classification Schema (5 classes):
%     1 - No contact:          No conductivity change
%     2 - Light touch:         Small force, medium area
%     3 - Firm press:          Large force, medium area
%     4 - Point contact:       Medium force, very small area
%     5 - Distributed contact: Medium force, large area
%
%   Parameters:
%     touch_type - String identifier: 'none', 'light', 'firm', 'point', 'distributed'
%     params     - Struct with fields: .radius, .conductivity, .x, .y
%
%   Returns:
%     class_id   - Integer class label (1-5)
%     class_name - Human-readable class name string
%     metadata   - Struct with all physical parameters for traceability
%
%   Example:
%     p = struct('radius', 0.08, 'conductivity', 0.90, 'x', 0.1, 'y', -0.2);
%     [id, name, meta] = encode_label('light', p);

    arguments
        touch_type (1,:) char {mustBeMember(touch_type, ...
            {'none', 'light', 'firm', 'point', 'distributed'})}
        params struct
    end

    % Class mapping
    class_map = struct( ...
        'none',        struct('id', 1, 'name', 'No contact'), ...
        'light',       struct('id', 2, 'name', 'Light touch'), ...
        'firm',        struct('id', 3, 'name', 'Firm press'), ...
        'point',       struct('id', 4, 'name', 'Point contact'), ...
        'distributed', struct('id', 5, 'name', 'Distributed contact'));

    class_id = class_map.(touch_type).id;
    class_name = class_map.(touch_type).name;

    % Metadata for full traceability
    metadata = struct( ...
        'class_id', class_id, ...
        'class_name', class_name, ...
        'touch_type', touch_type, ...
        'radius', params.radius, ...
        'conductivity', params.conductivity, ...
        'x', params.x, ...
        'y', params.y);
end