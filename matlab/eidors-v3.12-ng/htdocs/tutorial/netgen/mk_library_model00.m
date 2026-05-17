if exist('ng.opt','file')
    delete('ng.opt');
end
res = evalc('mk_library_model(''list'')');
res = regexprep(res,'<[^>]+>',''); %remove matlab links
res = horzcat(sprintf('>> mk_library_model(''list'')\n'), res);
writelines(res,'mk_library_model_list.txt');