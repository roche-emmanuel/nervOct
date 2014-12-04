function [network final_rms] = nnTrainNetworkCUDA(training,network,cfg)
% method used to train a network on the input dataset
% using the CUDA backpropagation implementation through the method train_bp

assert(exist('training', 'var') == 1, 'must provide a training structure')
assert(exist('network', 'var') == 1, 'must provide a network structure')

%  You should also try different values of lambda
rms_stop = training.rms_stop;
max_iter = training.max_iterations;

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

lsizes = network.layer_sizes;
initial_nn_params = network.weights;

% X
% X(:,1:7)
% yy
nt = numel(lsizes)-1;

% We should try to inject the current network weights into the bp:
% weights = network.weights;

% iweights = zeros(numel(weights),1);

% pos = 1;
% for i=1:nt,
% 	n = lsizes(i+1);
% 	m = lsizes(i)+1;
% 	count = n*m;
% 	mat = reshape(weights(pos:pos+count-1),n,m)';
% 	iweights(pos:pos+count-1) = mat(:);
% 	pos += count;
% end

% % Check that we can convert that back into the input weights:
% weights2 = zeros(numel(iweights),1);

% pos = 1;
% for i=1:nt,
% 	n = lsizes(i+1);
% 	m = lsizes(i)+1;
% 	count = n*m;
% 	mat = reshape(weights(pos:pos+count-1),m,n)';
% 	weights2(pos:pos+count-1) = mat(:);
% 	pos += count;
% end

% assert(sum(abs(weights-weights2))==0.0,'Invalid weight conversion process.')

% Call the training method here:
[weights final_rms] = train_bp(lsizes,X,yy,rms_stop,max_iter); %,iweights);

assert(all(isfinite(weights)),'Some computed weights are not finite!');

% once we have the weights, we must convert them from row major to column major data for octave usage:
cweights = zeros(numel(weights),1);

pos = 1;
for i=1:nt,
	n = lsizes(i+1);
	m = lsizes(i)+1;
	count = n*m;
	mat = reshape(weights(pos:pos+count-1),m,n)';
	cweights(pos:pos+count-1) = mat(:);
	pos += count;
end

network.weights = cweights;
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
%!	nn = nnInitNetwork([cfg.num_features 10 3],cfg);
%!	% tr.deep_training = true;
%!	tic();
%!	nn = nnTrainNetworkCUDA(tr,nn,cfg);
%!	toc();

% ==> Should exhibit speed improvement when training on more epochs:
%!test
%! 	% prepare the training set:
%!	cfg = config();
%!	tr = nnPrepareTraining(1:1,cfg);
%!	nn = nnInitNetwork([cfg.num_features 10 3],cfg);
%!	tr.max_iterations = 100;
%!	tic();
%!	nn = nnTrainNetworkCUDA(tr,nn,cfg);
%!	toc();

% ==> It should overfitting when we don't have enough data:
%!test
% prepare the training set:
%!	cfg = config();
%!	tr = nnPrepareTraining(1:1,cfg);
%!	nn = nnInitNetwork([cfg.num_features 20 3],cfg);
%!	tr.max_iterations = 1000;
%!	tr.train_ratio = 0.1;
%!	tr.rms_stop = 0.002;
%!	tic();
%!	[nn final_rms] = nnTrainNetworkCUDA(tr,nn,cfg);
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
%!	assert(abs(rms-final_rms)<1e-10,'Invalid result in RMS computation: %f!=%f\n',rms,final_rms)
