% 2D image $Id: tutorial022b.m 2701 2011-07-13 12:40:27Z aadler $

img= mk_image(mdl, conduc);
fsol= fwd_solve(img)
% fsol is the voltage drop across the resistor
