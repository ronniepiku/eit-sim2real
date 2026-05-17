% Reject electrodes + Image $Id: electrode_errors02.m 4101 2013-05-28 19:18:55Z bgrychtol $

imdl.meas_icov = meas_icov_rm_elecs( imdl, 13);

img = inv_solve(imdl, vh, vi(:,20));
show_fem(img,[0,1,0]); axis off
print_convert electrode_errors02a.png


imdl.meas_icov = meas_icov_rm_elecs( imdl, [13,5]);
img = inv_solve(imdl, vh, vi(:,20));
show_fem(img,[0,1,0]); 
print_convert electrode_errors02b.png
