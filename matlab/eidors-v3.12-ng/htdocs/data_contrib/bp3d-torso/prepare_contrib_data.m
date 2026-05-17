%PREPARE_CONTRIB_DATA Download and package bp3d-torso data

% (C) 2024 Bartek Grychtol. License: GPL v2 or v3
% $Id: prepare_contrib_data.m 7002 2024-11-24 13:11:35Z aadler $

unzip('https://zenodo.org/records/11047831/files/stl.zip');

attribution = ...
   ['"Torso Organ Meshes" ' ...
    '(https://doi.org/10.5281/zenodo.11047831) by Bartłomiej Grychtol ' ...
    '/ A derivative of "BodyParts3D" '...
    '(https://dbarchive.biosciencedbc.jp/en/bodyparts3d) by '...
    'The Database Center for Life Science.'];

license   =  ['Creative Commons Attribution-ShareAlike 2.1 Japan '...;
       '(http://creativecommons.org/licenses/by-sa/2.1/jp/deed.en_US)'];

%%
path = 'stl';
vessels = {'Aorta','IVC','SVC','PA', 'LSPV', 'LIPV','RSPV','RIPV'};

rmbnd = @(x) rmfield(x, 'boundary');
for cname = vessels
    tmp = struct();
    name = cname{1};
    tmp.inner = rmbnd(stl_read([path filesep  name ' inner.stl']));
    tmp.outer = rmbnd(stl_read([path filesep  name ' outer.stl']));
    tmp.ring  = rmbnd(stl_read([path filesep  name ' ring.stl']));
    tmp.cap   = rmbnd(stl_read([path filesep  name ' cap.stl']));
    eval(sprintf('%s = tmp; clear tmp;',name));
end

%%
organs = {'Heart shell', 'Blood right', 'Blood left', ...
            'Lung right', 'Lung left', 'Trachea'};
orgnames = genvarname(organs);

for i = 1:numel(organs)
    tmp = struct();
    name = organs{i};
    tmp = rmbnd(stl_read([path filesep  name '.stl']));
    eval(sprintf('%s = tmp; clear tmp;',orgnames{i}));
end
%%
save('bp3d_organs',vessels{:}, orgnames{:}, 'attribution', 'license')

%%
torso = rmbnd(stl_read([path filesep  'Skin torso.stl']));
save('bp3d_torso','torso', 'attribution', 'license');