% 2D models $Id: mk_common_model_2d_01.m 2791 2011-07-14 22:38:07Z aadler $

clf; subplot(221)

imdl= mk_common_model('a2c0',16);
show_fem(imdl.fwd_model);
print_convert  mk_common_model_2d_01a.png

imdl= mk_common_model('b2c0',16);
show_fem(imdl.fwd_model);
print_convert  mk_common_model_2d_01b.png

imdl= mk_common_model('c2c0',16);
show_fem(imdl.fwd_model);
print_convert  mk_common_model_2d_01c.png

imdl= mk_common_model('d2c0',16);
show_fem(imdl.fwd_model);
print_convert  mk_common_model_2d_01d.png
