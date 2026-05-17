torso = mk_thorax_model_bp3d('2x16_odd-even','fine');

%%
fmdl = torso.fwd_model;
clf
h = show_fem(fmdl);
ch = get(gca, 'Children');
for i = 1:numel(ch)
   set(ch(i),'EdgeColor','none');
end
set(h,'FaceColor',[.5 .5 .5]);
set(h,'FaceLighting','phong','AmbientStrength',0.7)
light('Position',[1 -.5 0.2],'Style','infinite');
set(h,'FaceAlpha',.3)

hold on

idx   = [   2,    3,    4];
alpha = [   .3,   .3,    1];
color = [  .3     0     0;
           .3     0     0;
           .75    0     0;];

for i = 1:numel(idx)
   obj = rmfield(fmdl,'boundary');
   obj.elems = obj.elems(fmdl.mat_idx{idx(i)},:);
   obj = fix_boundary(obj);
   oh(i) = show_fem(obj);
   set(oh(i),'EdgeColor','none')
   set(oh(i),'FaceColor',color(i,:),'FaceLighting','phong')
   set(oh(i),'FaceAlpha',alpha(i))
end

%oxygenated blood
idx = [5 9:12];
for i = 1:numel(idx)
   obj = rmfield(fmdl,'boundary');
   obj.elems = obj.elems(fmdl.mat_idx{idx(i)},:);
   obj = fix_boundary(obj);
   oh(i) = show_fem(obj);
   set(oh(i),'EdgeColor','none')
   set(oh(i),'FaceColor',[1 0 0],'FaceLighting','phong')
   
end

%de-oxygenated blood
idx = [6:8];
for i = 1:numel(idx)
   obj = rmfield(fmdl,'boundary');
   obj.elems = obj.elems(fmdl.mat_idx{idx(i)},:);
   obj = fix_boundary(obj);
   oh(i) = show_fem(obj);
   set(oh(i),'EdgeColor','none')
   set(oh(i),'FaceColor',[0 0 1],'FaceLighting','phong')
end
axis off
hold off
%%
view(3)
print_convert('torso_colors_01.jpg','-r600')
%%
view(0,0)
print_convert('torso_colors_02.jpg','-r600')

clf
torso.calc_colours.ref_level = 1;
torso.calc_colours.transparency_thresh = .01;
%show_3d_slices(torso, [1242],[],[]);
for i = [13 15 19:22]
   torso.elem_data(torso.fwd_model.mat_idx{i}) = 3;
end

[slc, p] = mdl_slice_mesher(torso, [inf inf 1242]);
slc.calc_colours.ref_level = 1;
slc.calc_colours.transparency_thresh = .01;
h = show_fem(slc);
set(h,'LineWidth',1)
ch = get(gca,'Children');
% need a custom colormap
idx = unique(get(ch(18), 'FaceVertexCData'));
color = [  .3     0     0; % lung
            1    .7    .7; % vessel
           .75    0     0; % heart wall
           0      0     1; % deox HB
           1      0     0; % oxHB 
            ];
c = ones(254,3);
for i = 1:numel(idx)
   c(idx(i),:) = color(i,:);
end
colormap(c);
view(2)
set(gca,'ydir','reverse')
p.FaceVertexCData(:) = 0;
h2 = patch(p);
set(h2,'FaceAlpha',0)
set(h2,'LineWidth',0.1)
set(h2,'EdgeAlpha',.8)
axis off
print_convert('torso_slice_01.jpg','-r600');
