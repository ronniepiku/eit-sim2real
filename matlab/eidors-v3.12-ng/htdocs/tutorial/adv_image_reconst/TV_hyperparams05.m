% Add noise $Id: TV_hyperparams05.m 1535 2008-07-26 15:36:27Z aadler $

sig= sqrt(norm(vi.meas - vh.meas));
m= size(vi.meas,1);
vi.meas = vi.meas + .01*sig*randn(m,1);
