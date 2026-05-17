function elem_centres = get_element_centers(fmdl)
%GET_ELEMENT_CENTERS Compute the centroid of each finite element.
%
%   elem_centres = GET_ELEMENT_CENTERS(fmdl) returns an (n_elems x n_dims)
%   matrix of element centroids for the forward model fmdl.
%
%   For 2D triangular elements (3 nodes per element) or 3D tetrahedral
%   elements (4 nodes per element), the centroid is the mean of the vertex
%   coordinates.

    nodes = fmdl.nodes;       % (n_nodes x n_dims)
    elems = fmdl.elems;       % (n_elems x nodes_per_elem)

    n_elems = size(elems, 1);
    n_dims = size(nodes, 2);
    nodes_per_elem = size(elems, 2);

    % Compute centroids by averaging vertex coordinates per element
    elem_centres = zeros(n_elems, n_dims);
    for d = 1:n_dims
        coords = reshape(nodes(elems, d), n_elems, nodes_per_elem);
        elem_centres(:, d) = mean(coords, 2);
    end
end