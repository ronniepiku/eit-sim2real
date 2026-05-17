% Simple distmesh model $Id: simple_2d_model01.m 2767 2011-07-14 16:32:47Z bgrychtol $

% Create small model with little mesh refinement
imdm=mk_common_model('a2d1c',16);
subplot(121)
show_fem(imdm.fwd_model);
axis on; axis equal; axis([-1.1 1.1 -1.1 1.1]);

% close in view
subplot(122)
show_fem(imdm.fwd_model);
axis equal; axis([-.1 .1 .85 1.05]);

print_convert simple_2d_model01a.png '-density 125'
