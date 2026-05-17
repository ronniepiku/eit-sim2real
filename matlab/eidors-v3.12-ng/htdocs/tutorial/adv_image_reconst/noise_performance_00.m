% both matlab and octave see as clear local
clear *
zipfilecontents = unzip('../../../build/external-tools/regularization-toolbox/regu.zip','regtools');
if exist('OCTAVE_VERSION')
   % Lots of these warnings from the regtools
   warning('off','Octave:possible-matlab-short-circuit-operator');
end

