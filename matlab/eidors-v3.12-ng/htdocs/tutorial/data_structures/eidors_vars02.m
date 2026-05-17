% Show EIDORS colours $Id: eidors_vars02.m 2757 2011-07-14 15:56:06Z bgrychtol $

subplot(131)
calc_colours('npoints',32);
show_slices(img);

subplot(132)
calc_colours('npoints',128);
show_slices(img);

subplot(133)
calc_colours('npoints',64);
show_slices(img); %default value
print_convert eidors_vars02a.png '-density 100'
