% $Id: mk_GREIT_matrix04.m 2475 2011-03-02 20:40:30Z aadler $

rimg = inv_solve(imdl,vh,vi); %Reconstruct

show_fem(rimg);
print_convert mk_GREIT_matrix04a.png '-density 60'
