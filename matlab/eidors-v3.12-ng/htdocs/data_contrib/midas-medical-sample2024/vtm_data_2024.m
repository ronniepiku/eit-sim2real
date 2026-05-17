pth = 'EITData2024_10_31_08_43_09__S01.vtm';
[dd,auxdata,stim]= eidors_readdata(pth);
dd = dd(:,1:end-100);
fmdl = mk_library_model('adult_male_16el');
fmdl.stimulation = stim;
fmdl.frame_rate = auxdata.frame_rate;

imdl = select_imdl(fmdl, {'GREIT:NF=0.5 32x32'});
imgr = inv_solve(imdl, mean(dd,2),dd);
imgr.elem_data = subtract_rank(imgr.elem_data,0.95);
[ROIs,imgR] = define_ROIs(imdl,[-2,0,2],[2,0,-2],struct('normalize',0));
sig = -ROIs*imgr.elem_data;
lt = {'LineWidth',2};
subplot(211);
plot(imgr.time,sig',lt{:}); box off; xlim tight;
 legend('RV','LV','RD','LD');
vertlines(50 + [1,50]*.5);
subplot(212);
imgr.get_img_data.frame_select = 500 + (1:50)*5;
imgr.calc_colours.ref_level = 0;
imgr.show_slices.img_cols = 17;
imgr.calc_colours.greylev =  .01;
imgr.calc_colours.backgnd =[1,1,1];
show_slices(imgr)

print_convert("vtm_data_2024a.jpg");
