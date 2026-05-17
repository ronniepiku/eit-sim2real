img = mk_image(fmdl, 1);
vh = fwd_solve(img);
esp = elem_select(fmdl, '(x-0.5).^2+(y-0.3).^2+(z-1.5).^2<0.2^2');
esn = elem_select(fmdl, '(x+0.5).^2+(y+0.3).^2+(z-2).^2<0.2^2');
img.elem_data = 1 + 0.01*esp - 0.01*esn;
vi = fwd_solve(img);

clf
show_fem(img)
print_convert GREIT3D_arbitrary_02a.jpg

clf
show_3d_slices(img,[1.5, 2],[-.5 .5],0) % cuts through the targets
print_convert GREIT3D_arbitrary_02b.jpg

clf
img.fwd_model.mdl_slice_mapper.npx = 64;
img.fwd_model.mdl_slice_mapper.npy = 32;
show_slices(img,5)
print_convert GREIT3D_arbitrary_02c.png