% Basic 3d model $Id: basic_3d_04.m 6511 2022-12-30 18:38:45Z aadler $

% Simulate Voltages and plot them
vh= fwd_solve(img1);
vi= fwd_solve(img2);

plot([vh.meas, vi.meas]);
axis tight
print_convert('basic_3d_04a.png','-density 60','-p10x5');
