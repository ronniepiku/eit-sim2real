% simulate homogeneous $Id: simulate_2d_test02.m 3273 2012-06-30 18:00:35Z aadler $

% stimulation pattern: adjacent
stim_pat= mk_stim_patterns(n_elec,1,'{ad}','{ad}',{},1);

smdl.stimulation= stim_pat;
himg= mk_image(smdl,1);

vh= fwd_solve(himg); vh= vh.meas;
