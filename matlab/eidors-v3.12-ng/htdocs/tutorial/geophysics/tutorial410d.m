% Show images $Id: tutorial410d.m 4088 2013-05-27 15:32:00Z bgrychtol $

subplot(121)
axis square; view(30.,80.);
show_fem(gallery_3D_img);

subplot(122)
gallery_3D_resist= gallery_3D_img; % Create resistivity image
gallery_3D_resist.elem_data= 1./gallery_3D_img.elem_data;
show_slices(gallery_3D_resist,[inf,inf,0]);

% print_convert tutorial410d.png;
