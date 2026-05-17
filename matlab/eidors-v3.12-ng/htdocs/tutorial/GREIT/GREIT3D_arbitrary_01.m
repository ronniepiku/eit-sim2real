
ellipsoid.center = [0;0;3];
ellipsoid.axis_a = [2;0;0];
ellipsoid.axis_b = [0;1;0];
ellipsoid.axis_c = [0;0;3.5];

ortho_brick.opposite_corner_a = [2; 1; 0];
ortho_brick.opposite_corner_b = [-2;-1; 3];
bg.intersection.ellipsoid = ellipsoid;
bg.intersection.ortho_brick = ortho_brick;

% shape
mdl = ng_mk_geometric_models(bg);
% add electrodes
fmdl = place_elec_on_surf(mdl,[12,0, 1.25, 1.75, 2.25],[0.1, 0, 0.02]);
% re-order electrodes and add stimulation
v = reshape(1:36,[],3)';
fmdl.electrode = fmdl.electrode(v(:));
fmdl.stimulation = mk_stim_patterns(36,1,[0 9],[0,5],{},1);

show_fem(fmdl);
print_convert GREIT3D_arbitrary01.jpg