% Distmesh models $Id: common_models03.m 2161 2010-04-04 20:33:46Z aadler $
imdl=mk_common_model('c2C1',16);
show_fem(imdl.fwd_model,[0,1]);

print_convert('common_models03a.png','-density 60');

imdl=mk_common_model('d2t3',16);
show_fem(imdl.fwd_model,[0,1]);

print_convert('common_models03b.png','-density 60');
