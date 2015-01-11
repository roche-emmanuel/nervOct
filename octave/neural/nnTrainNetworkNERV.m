function [network] = nnTrainNetworkNERV(training,network,cfg)
% method used to train a network on the input dataset
% using the NERV backpropagation implementation through the method nn_gradient_descent

assert(exist('training', 'var') == 1, 'must provide a training structure')
assert(exist('network', 'var') == 1, 'must provide a network structure')

X = training.X_train;
y = training.y_train;

% We might use only a part of the available training data:
ratio = training.train_ratio;
m = floor(0.5 + ratio*size(X,1));
X = X(1:m,:);

% Before continuing we have to select the target symbol we want to train on.
y = y(1:m,cfg.target_symbol_pair);

% Now we need to convert the label vector into a matrix:
yy = nnBuildLabelMatrix(y)';

% Prepare the training traits:
desc.lsizes = network.layer_sizes;
desc.X_train = X;
desc.y_train = yy;

size(network.weights)

desc.params = network.weights;
desc.epsilon = training.learning_rate;
desc.verbose = training.verbose;
desc.momentum = training.momentum;
desc.maxiter = training.max_iterations;
desc.evalFrequency = training.eval_frequency;
desc.miniBatchSize = training.mini_batch_size;
desc.lambda = training.regularization_param;
desc.minCostDecrease = training.min_cost_decrease;
desc.learningDecay = training.learning_decay;
desc.useSoftmax = training.with_softmax;

if isfield(training,'ping_frequency')
desc.pingFrequency = training.ping_frequency;
end

if isfield(training,'dropouts')
% fprintf('Setting up dropout...\n');
desc.dropouts = training.dropouts;
end

if training.early_stopping,
	Xcv = training.X_cv;
	% size(X)
	ycv = training.y_cv(:,cfg.target_symbol_pair);
	% size(y)
	yycv = nnBuildLabelMatrix(ycv)';
	
	desc.validationWindowSize = training.validation_window_size;
	desc.X_cv = Xcv;
	desc.y_cv = yycv;
end

[weights, costs, iters, Jcv] = nn_gradient_descent(desc);

if isfield(training,'dropouts')
% fprintf('Rescaling weights...\n');
weights = nnRescaleParameters(weights, desc.lsizes, desc.dropouts);
end

% pred_jcv = nnCostFunction(weights,desc.lsizes,desc.X_cv,desc.y_cv,0)
% pred_jcv = nn_cost_function_cuda(weights,desc.lsizes,desc.X_cv,desc.y_cv,0)
% jcv_error = abs(Jcv-pred_jcv)

network.weights = weights;
network.costs = costs;
network.cost_iters = iters;
network.Jcv = Jcv;

end

% ==> Must provide the training structure:
%!error <must provide a training structure> nnTrainNetworkNERV();

% ==> Must provide the network structure:
%!error <must provide a network structure> nnTrainNetworkNERV(rand(1,1));

% ==> Should be able to train a simple network:
%!test
%! 	%	prepare the training set:
%!	cfg = config();
%!	cfg.use_PCA = false;
%!	tr = nnPrepareTraining(1:1,cfg);
%!	tr.early_stopping = true;
%!	tr.max_iterations = 0;
%!	tr.learning_rate = 0.001;
%!	tr.mini_batch_size = 128;
%!	tr.validation_window_size = 50;
%!	tr.regularization_param = 10.0;
%!	tr.dropouts = [0.8, 0.5, 0.5, 0.5];
%!	%tr.dropouts = [0.8, 0.5];
%!	nn = nnInitNetwork([tr.num_features 512 128 32 3],cfg);
%!	%nn = nnInitNetwork([tr.num_features 32 3],cfg);
%!	tic();
%!	nn = nnTrainNetworkNERV(tr,nn,cfg);
%!	toc();
%!	% We should also chekc that the Jcv value is correct:
%!	ev = nnEvaluateNetwork(tr,nn,cfg);
%!	assert(abs(nn.Jcv - ev.J_cv)<1e-10,'Mistmatch in Jcv computation: %f!=%f',nn.Jcv,ev.J_cv);
%!	ev.J_train
%!	ev.J_cv
%!	ev.accuracy_train
%!	ev.accuracy_cv
%! 	% Now we can draw the evolution of the costs:
%!	figure; hold on;
%!	h = gcf();	
%!	%nn.cost_iters;
%!	%nn.costs;
%!	plot(nn.cost_iters, nn.costs, 'LineWidth', 2, 'Color','b');
%!	legend('Jcv');
%!	title('Learning progress');
%!	xlabel('Number of epochs');
%!	ylabel('Cv Cost');
%!	hold off;
