% Create 2D model of a tunnel $Id: tunnelsim05.m 2361 2010-11-08 10:50:48Z aadler $ 

extra={'ball', 'solid ball = sphere(0,0,0;1) -maxh=0.25;'};
cmdl= ng_mk_cyl_models([0,15,3],[0],[0.1,0,0.05],extra);

show_fem(cmdl);                      print_convert tunnelsim05a.png
show_fem(cmdl); axis(2*[-1,1,-1,1]); print_convert tunnelsim05b.png
