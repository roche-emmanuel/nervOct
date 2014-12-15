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
yy = nnBuildLabelMatrix(y);


% Prepare the training traits:
desc.lsizes = network.layer_sizes;
desc.X_train = X;
desc.y_train = yy;
desc.params = network.weights;
desc.epsilon = 0.00001;
desc.momentum = 0.99;
desc.maxiter = training.max_iterations;
desc.evalFrequency = 32;
desc.miniBatchSize = 32;
desc.lambda = training.regularization_param;
if isfield(training,'dropouts')
fprintf('Setting up dropout...\n');
desc.dropouts = training.dropouts;
end

if training.early_stopping,
	Xcv = training.X_cv;
	% size(X)
	ycv = training.y_cv(:,cfg.target_symbol_pair);
	% size(y)
	yycv = nnBuildLabelMatrix(ycv);
	
	desc.validationWindowSize = 100;
	desc.X_cv = Xcv;
	desc.y_cv = yycv;
end

[weights, costs, iters, Jcv] = nn_gradient_descent(desc);

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
%!	tr.dropouts = [0.8, 0.5, 0.5, 0.5];
%!	nn = nnInitNetwork([tr.num_features 512 128 32 3],cfg);
%!	tic();
%!	nn = nnTrainNetworkNERV(tr,nn,cfg);
%!	toc();
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
