% 2D models $Id: mk_common_model_2d_03.m 2791 2011-07-14 22:38:07Z aadler $

clf; subplot(221)

imdl= mk_common_model('c2t2',16);
show_fem(imdl.fwd_model);
print_convert mk_common_model_2d_03a.png

imdl= mk_common_model('b2t3',8);
show_fem(imdl.fwd_model);
print_convert mk_common_model_2d_03b.png

imdl= mk_common_model('b2T2',8);
show_fem(imdl.fwd_model);
print_convert mk_common_model_2d_03c.png

imdl= mk_common_model('c2T3',16);
show_fem(imdl.fwd_model);
print_convert mk_common_model_2d_03d.png
