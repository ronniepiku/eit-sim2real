% Simulate data $Id: dual_model03.m 2162 2010-04-04 20:49:45Z aadler $

for model= 1:3
   img= inv_solve(imdl(model), vh, vi);
   subplot(2,3,model)
   show_fem(img);
end

print_convert dual_model03a.png;
