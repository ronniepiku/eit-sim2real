% Solve resistor model
% $Id: tutorial020c.m 3127 2012-06-08 16:19:25Z bgrychtol $

% Now we complete the fwd_model
r_mdl.jacobian= @jacobian_perturb;

% Now create an inverse model
i_mdl= eidors_obj('inv_model','resistor inverse');
i_mdl.fwd_model= r_mdl;
i_mdl.jacobian_bkgnd.value= 1000;

% regulatization not needed for this problem
i_mdl.RtR_prior= @prior_tikhonov;
i_mdl.hyperparameter.value= 0;

i_mdl.reconst_type= 'difference';
i_mdl.solve= @inv_solve_diff_GN_one_step;

% Reconstruct resistor change
reconst= inv_solve(i_mdl, data_1k0, data_1k2);
