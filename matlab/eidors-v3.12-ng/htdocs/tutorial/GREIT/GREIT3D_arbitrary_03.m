% Setup a 3D rec_model 
vopt.cube_voxels = true;
%vopt.imgsz = [64, 64, 64]; % looks nice, but takes very long
vopt.imgsz = [48, 48, 48]; % A bit faster
vopt.downsample = [2, 1]; % reduce density of targets
vopt.save_memory = 1;
[imdl, distr] = GREIT3D_distribution(fmdl,vopt);

% limit training points to where the electrodes are
distr(:,(distr(3,:)>2.5)) = [];
distr(:,distr(3,:)<1) = [];
gopt.distr = distr;

gopt.noise_figure = 1.0;

% train GREIT
imdl3 = mk_GREIT_model(imdl, 0.2, [], gopt);
