% Human breathing $Id: demo_algs03.m 1559 2008-07-27 03:19:58Z aadler $

% Electrodes on back
load montreal_data_1995
v(4).vh = double( zc_resp(idx,1) );
v(4).vi = double( zc_resp(idx,22) );
