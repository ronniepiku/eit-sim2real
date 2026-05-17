% load RPI tank data $Id: rpi_data02a.m 2236 2010-07-04 14:16:21Z aadler $

load Rensselaer_EIT_Phantom;
vh = real(ACT2006_homog);
vi = real(ACT2000_phant);

if 1
% Preprocessing data. We ensure that each voltage sequence sums to zero
  for i=0:30
    idx = 32*i + (1:32);
    vh(idx) = vh(idx) - mean(vh(idx));
    vi(idx) = vi(idx) - mean(vi(idx));
  end
end
