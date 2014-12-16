function [J_train, J_cv, m_vals] = nnComputeLearningCurves(ratios,plsizes,training,cfg)
% Compute the cost values for all the ratios provided.

num = numel(ratios);
J_train = zeros(num,1);
J_cv = zeros(num,1);
m_vals = zeros(num,1);

m = size(training.X_train,1);

ntrials = cfg.default_num_random_trials;

for i=1:num,
	training.train_ratio = ratios(i);
	
	nn = nnSelectRandomNetwork(ntrials,plsizes,training,cfg);

	% nn = nnInitNetwork(lsizes,cfg);
	
	% if cfg.use_CUDA
	% 	nn = nnTrainNetworkCUDA(training,nn,cfg);
	% else
	% 	nn = nnTrainNetwork(training,nn,cfg);
	% end
	
	ev = nnEvaluateNetwork(training,nn,cfg);
	num_m = floor(0.5 + ratios(i)*m)
	if cfg.verbose
		ev.J_train;
		ev.J_cv;
		ev.accuracy_train
		ev.accuracy_cv
	end
	m_vals(i) = num_m;
	J_train(i) = ev.J_train;
	J_cv(i) = ev.J_cv;
end

end

% ==> Should compute the cost values:
%!test
%!	cfg = config();
%!	cfg.use_PCA = false;
%!	tr = nnPrepareTraining(1:3,cfg);
%!	tr.max_iterations = 0;
%!	nstep=20;
%!	tic();
%!	[jtrain, jcv, m_vals] = nnComputeLearningCurves((1:nstep)/nstep,[30 3],tr,cfg)
%!	%[jtrain, jcv, m_vals] = nnComputeLearningCurves((1:nstep)/nstep,[128 64 32 3],tr,cfg)
%!	toc();
%!	h = plotLearningCurves(m_vals,jtrain,jcv);
%!  saveas(h,cfg.learning_curves_graph_file);
