% Simulate data $Id: dual_model05.m 2162 2010-04-04 20:49:45Z aadler $

% Reconstruct
for model= 1:3
   img(model)= inv_solve(imdl(model), vh, vi);
end

% Show image mapped to fine model
for model= 1:3
   subplot(2,3,model)
   show_fem(img(model));
end

print_convert dual_model05b.png;

% Show image mapped to coarse model
for model= 1:3
   subplot(2,3,model)
   img(model).fwd_model = mdl_coarse.fwd_model;
   show_fem(img(model));
end

print_convert dual_model05a.png;
