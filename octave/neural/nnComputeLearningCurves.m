function [J_train, J_cv, m_vals] = nnComputeLearningCurves(ratios,lsizes,training,cfg)
% Compute the cost values for all the ratios provided.

num = numel(ratios);
J_train = zeros(num,1);
J_cv = zeros(num,1);
m_vals = zeros(num,1);

m = size(training.X_train,1);


nn = nnInitNetwork(lsizes);

for i=1:num,
	training.train_ratio = ratios(i);
	nn = nnTrainNetwork(training,nn,cfg);
	ev = nnEvaluateNetwork(training,nn,cfg);
	num_m = floor(0.5 + ratios(i)*m)
	ev.J_train
	ev.J_cv
	ev.accuracy_train
	ev.accuracy_cv
	m_vals(i) = num_m;
	J_train(i) = ev.J_train;
	J_cv(i) = ev.J_cv;
end

end

% ==> Should compute the cost values:
%!test
%!	cfg = config();
%!	cfg.default_max_training_iterations = 10000;
%!	tr = nnPrepareTraining(1:1,cfg);
%!	[jtrain, jcv, m_vals] = nnComputeLearningCurves((1:10)/10,[cfg.num_features 10 3],tr,cfg)
%!	plotLearningCurves(m_vals,jtrain,jcv);
