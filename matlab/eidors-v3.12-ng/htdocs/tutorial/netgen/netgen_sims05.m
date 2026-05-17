% Netgen simulation $Id: netgen_sims05.m 2787 2011-07-14 21:52:22Z bgrychtol $

subplot(211)
plot([vh.meas, vi.meas, vi.meas-vh.meas]);
print_convert netgen_sims05a.png '-density 100'
