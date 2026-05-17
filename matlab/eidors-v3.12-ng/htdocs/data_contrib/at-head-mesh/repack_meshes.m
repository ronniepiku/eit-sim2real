files = dir('*.7z');
curdir = cd;
for f = 1:numel(files)
   [status,cmdout] = system(['"C:\Program Files\7-Zip\7z.exe" x -aoa -o"',curdir,'" ',files(f).name])
end
mfiles = dir('*.mat');
for m = 1:numel(mfiles)
   S = load(mfiles(m).name);
   save(mfiles(m).name, '-struct','S', '-v7');
end