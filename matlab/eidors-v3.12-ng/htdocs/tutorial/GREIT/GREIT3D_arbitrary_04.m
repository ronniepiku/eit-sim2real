% reconstruct
rimg = inv_solve(imdl3, vh, vi);

clf
show_fem(rimg)
print_convert GREIT3D_arbitrary_04a.jpg

clf
show_3d_slices(rimg,[1.5, 2],[-.5 .5],0) % cuts through the targets
print_convert GREIT3D_arbitrary_04b.jpg

clf
show_slices(rimg,5)
print_convert GREIT3D_arbitrary_04c.png