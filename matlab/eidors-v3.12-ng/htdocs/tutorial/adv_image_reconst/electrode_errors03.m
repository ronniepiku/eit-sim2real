% Reciprocity error + Image $Id: electrode_errors03.m 4101 2013-05-28 19:18:55Z bgrychtol $

imdl.calc_reciproc_error.tau = 3e-4;
imdl.meas_icov = calc_reciproc_error( imdl, vi );

img = inv_solve(imdl, vh, vi(:,20)); show_fem(img,[0,1,0]);axis off
print_convert electrode_errors03a.png

imdl.calc_reciproc_error.tau = 3e-2;
imdl.meas_icov = calc_reciproc_error( imdl, vi );

img = inv_solve(imdl, vh, vi(:,20)); show_fem(img,[0,1,0]);axis off
print_convert electrode_errors03b.png
