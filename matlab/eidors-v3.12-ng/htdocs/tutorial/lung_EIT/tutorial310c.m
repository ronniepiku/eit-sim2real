% Lung images
% $Id: tutorial310c.m 4839 2015-03-30 07:44:50Z aadler $

load montreal_data_1995
imdl_d.hyperparameter.value=.2;
img= inv_solve(imdl_d, zc_resp(:,1), zc_resp);

clf; show_slices(img)
axis equal

print_convert tutorial310c.png;
