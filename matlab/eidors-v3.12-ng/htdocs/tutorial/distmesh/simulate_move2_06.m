% Reconstruct images $Id: simulate_move2_06.m 2781 2011-07-14 21:06:54Z bgrychtol $

imdl= mk_common_model('c2c2',16);
img= inv_solve(imdl, vh, vi);
subplot(121); show_fem(img); axis image
subplot(122); show_slices(img)

print_convert simulate_move2_06a.png '-density 125'
