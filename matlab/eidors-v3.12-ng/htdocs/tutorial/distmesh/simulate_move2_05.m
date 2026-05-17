% Create animated graphics $Id: simulate_move2_05.m 6448 2022-12-02 12:19:15Z aadler $

% Trim images
system('find -name "simulate_move2_04a*.png" -exec convert  -trim "{}" PNG8:"{}" ";"')

% Convert to animated Gif
system('convert -delay 50 simulate_move2_04a*.png -loop 0 simulate_move2_05a.gif')
