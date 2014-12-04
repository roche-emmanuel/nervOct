% ==> Should be able to train a simple network with early stopping:
%!test
%!	cfg = config();
%! 	%	load training set:
%!	fname = [cfg.datapath '/training_week1_pca.mat'];
%!	load(fname);
%!	nn = nnInitNetwork([tr.num_features 100 10 3],cfg);
%!	tic();
%!	nn = nnTrainNetworkCUDA(tr,nn,cfg);
%!	toc();
