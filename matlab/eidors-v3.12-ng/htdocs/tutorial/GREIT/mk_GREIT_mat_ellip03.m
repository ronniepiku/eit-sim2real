% $Id: mk_GREIT_mat_ellip03.m 2480 2011-03-03 20:43:07Z aadler $

for k=1:4
   rimg = inv_solve(imdl(k),vh,vi); %Reconstruct

   subplot(1,4,k)
   show_fem(rimg); axis square;
end

print_convert mk_GREIT_matrix03.png '-density 180'
