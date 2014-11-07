function [y] = nnPredict(network, X)
% Given an input neural network and a feature matrix,
% This method will compute the predictions obtained by the network.

% Retrieve the layer sizes from the network structure:
layer_sizes = network.layer_sizes;
nn_params = network.weights;

% Check how may layers we have:
nl = numel(layer_sizes);

% when we have nl layers, we have nl-1 weights matrices.
% That we should rebuild here:
nt = nl-1;

% Reshape nn_params back into the parameters Thetas i, the weight matrices for our neural network:
Thetas = cell(1,nt);

% numel(nn_params)

pos = 1;
for i=1:nt,
	n = layer_sizes(i+1);
	m = layer_sizes(i)+1;
	count = n*m;
	Thetas{1,i} = reshape(nn_params(pos:pos+count-1),n,m);
	pos += count;
end

% note that the pos variable is now on the next index available:
if pos ~= (size(nn_params,1)+1),
	error('Mismatch in unrolled vector size %d != %d',pos, size(nn_params,1)+1);
end

% Setup some useful variables
m = size(X, 1);

% First we need to compute h(x) for each example.
a = X';

for i=1:nt,
	% First we add the intercept term as a top row on the current activation:
	a = [ones(1,m); a];

	% Then we can compute the total input for the next layer:
	z = Thetas{1,i} * a;

	% Compute the output of the next layer:
	a = sigmoid(z);
end

% Now the final result is stored in a, the activation vector (eg. output) of the last layer.
% Thus we can rename this output as hx:
hx = a;

% hx is the resulting predicting where each colunm is for a given sample.
% We convert that into a matrix where is row is for a given sample:
yy = hx';

% Then we need to take the max on eash row. And we are only interested in the index corresponding to
% the maximum value on a given row:
[dummy, y] = max(yy, [], 2);

% Also, keep in mind that the index returned by Octave is 1-based,
% whereas, by conventions, the labels we use here are 0-based, thus we need to remove 1:
y = y-1;

end

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

% ==> It should be able to compute some predictions:
%!test
%!	cfg = config();
%!	cfg.default_max_training_iterations = 1000;
%!	cfg.default_deep_training = true;
%!	num_features = 20;
%!	num_labels = 3;
%!	num_samples = 30;
%!	tr = nnPrepareDebugTraining(num_samples,num_features,num_labels,cfg);
%!	nn = nnInitNetwork([num_features 6 num_labels]);
%!	% tr.deep_training = true;
%!	%for i=1:10,
%!		nn = nnTrainNetwork(tr,nn,cfg);
%!	%end
%!	pred = nnPredict(nn,tr.X_train);
%!	assert(size(pred,1)==size(tr.X_train,1),'Invalid number of rows')
%!	assert(size(pred,2)==1,'Invalid number of cols')
%!	y = tr.y_train(:,cfg.target_symbol_pair);
%!	% We should normally be able to predict perfectly the labels (?)
%!	err = mean( pred ~= y);
%!	assert(err==0,'Mismatch in origanl labels and prediction: error ratio=%f',err)

