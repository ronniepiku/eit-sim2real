% Show EIDORS colours $Id: eidors_vars01.m 3273 2012-06-30 18:00:35Z aadler $

imdl= mk_common_model('a2c2',16);
img= mk_image(imdl.fwd_model,0);
img.elem_data(1:2)=[1,1];
img.elem_data([14,16])=[-1,-1];
img.elem_data([27,30])=.5*[1,1];
show_fem(img);

print_convert  eidors_vars01a.png '-density 50'
