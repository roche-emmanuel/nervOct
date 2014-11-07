function nndata = nnInitNetwork(lsizes)
% method used to initialize a neural network structure

assert(exist('lsizes', 'var') == 1, 'must provide a layer size vector')

% save the laer sizeq as part of the configuration:
nndata.layer_sizes = lsizes;

% Prepare the network weights:
% Number of layers to consider for the test:
nl = numel(lsizes);

nn_params = [];
nt = nl-1;

% We generate some 'random' test data
for i=1:nt,
	theta = nnRandInitializeWeights(lsizes(i), lsizes(i+1));
	nn_params = [nn_params; theta(:)];
end

% save the weights:
nndata.weights = nn_params;
nndata.training_costs = [];  %By default the training cost should be empty.
end

% ==> Must provide the layer size:
%!error <must provide a layer size vector> nnInitNetwork();

% ==> Should have the expected number of parameters:

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

%!test
%!	num=10;
%!	for i=1:num,
%!		nl = random2(3,5);
%!		sizes = zeros(nl,1);
%!		for j=1:nl,
%!			sizes(j) = random2(3,6);
%!		end
%!		% compute the expected number of params:
%!		np = 0;
%!		for i=1:nl-1,
%!			np += (sizes(i)+1)*sizes(i+1);
%!		end
%!	
%!		nn = nnInitNetwork(sizes);
%!		nn.weights;
%!		assert(numel(nn.weights)>0,'No element for weights!')
%!		assert(numel(nn.weights)==np,'Invalid number of NN parameters %d != %d\n',numel(nn.weights),np);
%!	end

