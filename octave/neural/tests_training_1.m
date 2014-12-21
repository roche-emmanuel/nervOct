% ==> Should be able to train a simple network with early stopping:
%test
%	cfg = config();
% 	%	load training set:
%	fname = [cfg.datapath '/training_week1_pca.mat'];
%	load(fname);
%	nn = nnInitNetwork([tr.num_features 100 10 3],cfg);
%	tic();
%	nn = nnTrainNetworkCUDA(tr,nn,cfg);
%	toc();

%!test
%!	cfg = config();
%!	fname = [cfg.datapath '/training_weeks_1_1.mat'];
%!	load(fname);
%!	tr.early_stopping = true;
%!	tr.max_iterations = 0;
%!	tr.regularization_param = 0.0;

%!	%tr.dropouts = [0.8, 0.5, 0.5, 0.5];
%!	%tr.dropouts = [0.8, 0.5];

%!	tr.y_train(1:20,:)

%!	%nn = nnInitNetwork([tr.num_features 512 128 32 3],cfg);
%!	nn = nnInitNetwork([tr.num_features 512 3],cfg);
%!	%nn = nnInitNetwork([tr.num_features 32 3],cfg);
%!	nn = nnTrainNetworkNERV(tr,nn,cfg);
%!  %nn = nnTrainNetwork(tr,nn,cfg);
%!	nn.mu = tr.mu;
%!	nn.sigma = tr.sigma;
%!	%fname = [cfg.datapath '/nn_512_128_32_3_drop_weeks_1_4.mat'];
%!	%save('-binary',fname,'nn');
%!	nnEvaluateNetwork(tr,nn,cfg)

