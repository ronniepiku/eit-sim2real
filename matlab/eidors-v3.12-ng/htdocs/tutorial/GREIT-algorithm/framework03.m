% image prior $Id: framework03.m 1545 2008-07-26 17:33:46Z aadler $
load GREIT_Jacobian_ng_mdl_fine J map

% Remove space outside FEM model
J= J(:,map);
% inefficient code - but for clarity
diagJtJ = diag(J'*J) .^ (0.7);

R= spdiags( diagJtJ,0, length(diagJtJ), length(diagJtJ));

save ImagePrior_diag_JtJ R
