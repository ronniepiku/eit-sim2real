% Basic GREIT reconst $Id: electrode_errors05.m 4101 2013-05-28 19:18:55Z bgrychtol $

imgr= inv_solve(imdl, zc_resp(:,1), zc_resp(:,22));
imgr.calc_colours.ref_level=0; show_fem(imgr,[0,1]);

axis equal; axis([-1,1.05,-.8,.8]);
print_convert electrode_errors05a.png
