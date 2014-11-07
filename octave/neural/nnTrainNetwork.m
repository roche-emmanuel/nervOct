function [network] = nnTrainNetwork(training,network,cfg)
% method used to train a network on the input dataset

assert(exist('training', 'var') == 1, 'must provide a training structure')
assert(exist('network', 'var') == 1, 'must provide a network structure')

% Retrieve the max iteration number from the training structure
options = optimset('MaxIter', training.max_iterations);

%  You should also try different values of lambda
lambda = training.regularization_param;
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

% Keep in mind here that we need to transpose the y matrix:
yy = yy';

% size(y)
% size(X)

lsizes = network.layer_sizes;
initial_nn_params = network.weights;

assert(numel(initial_nn_params)>0,'No element for initial nn params!')

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, lsizes, X, yy, lambda);

% compute the initial cost:
% ic = costFunction(initial_nn_params)

count = 0;
while true,
	count++;
	fprintf('Performing training session %d...\n', count);

	[nn_params, cost, icount] = fmincg(costFunction, network.weights, options);
	% assert(size(cost,1)==icount,'Mismatch between number of iterations and number of cost values: %d != %d', size(cost,1),icount)
	
	% save the computed values:
	network.weights = nn_params;
	network.training_costs = [network.training_costs; cost];

	if !training.deep_training || (icount < training.max_iterations),
		break;
	end
end

end

% ==> Must provide the training structure:
%!error <must provide a training structure> nnTrainNetwork();

% ==> Must provide the network structure:
%!error <must provide a network structure> nnTrainNetwork(rand(1,1));

% ==> Should be able to train a simple network:
%!test
% prepare the training set:
%!	cfg = config();
%!	tr = nnPrepareTraining(1:1,cfg);
%!	nn = nnInitNetwork([cfg.num_features 10 3]);
%!	% tr.deep_training = true;
%!	nn = nnTrainNetwork(tr,nn,cfg);
%!	n = size(nn.training_costs,1);
%!	assert(n==tr.max_iterations,'Invalid number of training costs: %d',n)
