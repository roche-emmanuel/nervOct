function [perfs] = nnEvaluateNetwork(training,network,cfg)
% Method used to evaluate the performances of a network

assert(exist('training', 'var') == 1, 'must provide a training structure')
assert(exist('network', 'var') == 1, 'must provide a network structure')
assert(exist('cfg', 'var') == 1, 'must provide global config')


% First we compute the cost on the train dataset:
X = training.X_train;
y = training.y_train;

% We might use only a part of the available training data:
ratio = training.train_ratio;
m = floor(0.5 + ratio*size(X,1));

X = X(1:m,:);
y = y(1:m,cfg.target_symbol_pair);

% Current regularization parameter:
% NOTE: when evaluation the network we should not consider any regularization parameter.
% lambda = training.regularization_param;
lambda = 0;

yy = nnBuildLabelMatrix(y)';

% We should also compute the cost
perfs.J_train = nnCostFunction(network.weights, network.layer_sizes, X, yy, lambda);

% if !isfinite(perfs.J_train)
% 	X
% 	yy
% 	lambda
% 	network.layer_sizes
% 	error('Could not compute j_train with previous values.');
% end

% Compute the train accuracy:
% eg. The number of incorrectly classified samples:
pred = nnPredict(network, X);

perfs.accuracy_train = 1.0 - mean(double(pred~=y));

% Now compute the cost on the cross validation dataset:
X = training.X_cv;
% size(X)
y = training.y_cv(:,cfg.target_symbol_pair);
% size(y)
yy = nnBuildLabelMatrix(y)';

perfs.J_cv = nnCostFunction(network.weights, network.layer_sizes, X, yy, lambda);

% Compute the cv accuracy:
% eg. The number of incorrectly classified samples:
pred = nnPredict(network, X);

perfs.accuracy_cv = 1.0 - mean(double(pred~=y));

% Chekc how many times we predict a buy:
pred_none_count = sum(pred==0)
pred_buy_count = sum(pred==1)
pred_sell_count = sum(pred==2)

real_none_count = sum(y==0)
real_buy_count = sum(y==1)
real_sell_count = sum(y==2)

% Now we get the predictions of the network on those features:
% pred = nnPredict(network, X);

% Then we need to compare the 

end

% ==> Must provide the training structure:
%!error <must provide a training structure> nnEvaluateNetwork();

% ==> Must provide the network structure:
%!error <must provide a network structure> nnEvaluateNetwork(rand(1,1));

% ==> Must provide the global config:
%!error <must provide global config> nnEvaluateNetwork(rand(1,1),rand(1,1));

% ==> Should provide the J_train value:
%!test
%!	cfg = config();
%!	cfg.default_max_training_iterations = 10;
%!	tr = nnPrepareTraining(1:1,cfg);
%!	nn = nnInitNetwork([tr.num_features 10 3],cfg);
%!	% tr.deep_training = true;
%!	nn = nnTrainNetwork(tr,nn,cfg);
%!	ev = nnEvaluateNetwork(tr,nn,cfg);
%!	j1 = ev.J_train
%!	jcv1 = ev.J_cv
%!	assert(j1>0,'Invalid value for J_train')
%!	assert(jcv1>0,'J_cv should be bigger that 0')
%! 	% If we train again, then the cost should become lower:
%!	nn = nnTrainNetwork(tr,nn,cfg);
%!	ev = nnEvaluateNetwork(tr,nn,cfg);
%!	j2 = ev.J_train
%!	jcv2 = ev.J_cv
%!	assert(j2<j1,'train cost should decrease')
%!	assert(jcv2>0,'cv cost should be positive')
%!  ac_train = ev.accuracy_train
%!  ac_cv = ev.accuracy_cv

