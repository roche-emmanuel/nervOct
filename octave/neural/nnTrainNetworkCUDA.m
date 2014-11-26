function [network] = nnTrainNetworkCUDA(training,network,cfg)
% method used to train a network on the input dataset
% using the CUDA backpropagation implementation through the method train_bp

assert(exist('training', 'var') == 1, 'must provide a training structure')
assert(exist('network', 'var') == 1, 'must provide a network structure')

max_iter = training.max_iterations;
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
yy = nnBuildLabelMatrix(y)';

lsizes = network.layer_sizes;

% Here we need to check if we requested early stopping.
if training.early_stopping,
	% we need to train as long as we can reduce the error on the cv dataset:
	% To achieve this we keep track of the previous weights generated so far with their 
	% corresponding cost on cv:
	costs = [];
	wmat = [];

	Xcv = training.X_cv;
	% size(X)
	ycv = training.y_cv(:,cfg.target_symbol_pair);
	% size(y)
	yycv = nnBuildLabelMatrix(ycv)';

	% window_size is the number of cost we want to keep to compute the mean value
	window_size = 5; 
	max_wrong = 5;
	best_cost = -1;
	best_weights = [];
	cur_mean = 0.0;
	wrong_count = 0;

	iter_count = 0;

	weights = network.weights;

	while true
		weights = nn_cg_train(lsizes, X, yy, weights, lambda, max_iter);
		% evaluate those parameters:
		% Now compute the cost on the cross validation dataset:
		% J_cv = nnCostFunction(weights, lsizes, Xcv, yycv, lambda);
		J_train = nn_cost_function_cuda(weights, lsizes, X, yy, lambda);
		J_cv = nn_cost_function_cuda(weights, lsizes, Xcv, yycv, lambda);
		iter_count += max_iter;

		fprintf('Costs at iteration %d: J_train=%f, J_cv=%f\n',iter_count,J_train,J_cv)

		% keep the best weights so far:
		if best_cost < 0.0 || J_cv < best_cost,
			best_cost = J_cv;
			best_weights = weights;
		end

		% push this cost in the list:
		costs = [costs; J_cv];

		if size(costs,1)>window_size,
			costs = costs(2:end);
			% compute the new mean:
			cmean = mean(costs);
			if cmean > cur_mean,
				% increment the wrong count:
				wrong_count += 1;
				if wrong_count >= max_wrong,
					fprintf('Stopping with best cost %f\n',best_cost)
					network.weights = best_weights;
					break;
				end
			else
				% reset the wrong count:
				wrong_count = 0;
			end
		end

		% initialization phase:
		cur_mean = mean(costs);
	end
else
	% just compute the weights once:
	% fprintf('Performing training...\n')
	network.weights = nn_cg_train(lsizes, X, yy, network.weights, lambda, max_iter);
	% fprintf('Training done.\n')
end

end

% ==> Must provide the training structure:
%!error <must provide a training structure> nnTrainNetwork();

% ==> Must provide the network structure:
%!error <must provide a network structure> nnTrainNetwork(rand(1,1));

% ==> Should be able to train a simple network:
%!test
%! 	%	prepare the training set:
%!	cfg = config();
%!	tr = nnPrepareTraining(1:1,cfg);
%!	nn = nnInitNetwork([tr.num_features 10 3]);
%!	% tr.deep_training = true;
%!	tic();
%!	nn = nnTrainNetworkCUDA(tr,nn,cfg);
%!	toc();

% ==> Should exhibit speed improvement when training on more epochs:
%!test
%! 	% prepare the training set:
%!	cfg = config();
%!	tr = nnPrepareTraining(1:1,cfg);
%!	nn = nnInitNetwork([tr.num_features 10 3]);
%!	tr.max_iterations = 100;
%!	tic();
%!	nn = nnTrainNetworkCUDA(tr,nn,cfg);
%!	toc();

% ==> It should overfitting when we don't have enough data:
%!test
% prepare the training set:
%!	cfg = config();
%!	tr = nnPrepareTraining(1:1,cfg);
%!	nn = nnInitNetwork([tr.num_features 20 3]);
%!	tr.max_iterations = 1000;
%!	tr.train_ratio = 0.1;
%!	tr.rms_stop = 0.002;
%!	tic();
%!	nn = nnTrainNetworkCUDA(tr,nn,cfg);
%!	toc();
%! 	% Now check the RMS result:
%!	X = tr.X_train;
%!	y = tr.y_train;
%!	
%!	% We might use only a part of the available training data:
%!	ratio = tr.train_ratio;
%!	m = floor(0.5 + ratio*size(X,1));
%!	X = X(1:m,:);
%!		
%!	% Before continuing we have to select the target symbol we want to train on.
%!	y = y(1:m,cfg.target_symbol_pair);
%!	yy = nnBuildLabelMatrix(y);
%!
%!	% Note that here we need to use the initial prediction with 3 outputs to properly
%!	% compute the RMS value:
%!	[dumm pred] = nnPredict(nn, X);
%!	rms = (pred-yy) .* (pred-yy);
%!	rms = sqrt(sum(sum(rms))/numel(rms))/2.0;
