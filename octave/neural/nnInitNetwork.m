function nndata = nnInitNetwork(lsizes,cfg)
% method used to initialize a neural network structure

assert(exist('lsizes', 'var') == 1, 'must provide a layer size vector')
assert(exist('cfg', 'var') == 1, 'must provide a config')

% save the laer sizeq as part of the configuration:
nndata.layer_sizes = lsizes;

% Prepare the network weights:
% Number of layers to consider for the test:
nl = numel(lsizes);

nn_params = [];
nt = nl-1;

% We generate some 'random' test data
for i=1:nt,
	if cfg.use_sparse_init
		theta = nnSparseInitializeWeights(lsizes(i), lsizes(i+1), cfg.sparse_init_lot_size);
		nn_params = [nn_params; theta(:)];
	else
		theta = nnRandInitializeWeights(lsizes(i), lsizes(i+1));
		nn_params = [nn_params; theta(:)];
	end
end

% save the weights:
nndata.weights = nn_params;
end

% ==> Must provide the layer size:
%!error <must provide a layer size vector> nnInitNetwork();

% ==> Must provide the config:
%!error <must provide a config> nnInitNetwork([1 2 3]);

% ==> Should have the expected number of parameters:

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

%!test
%!	num=10;
%!	cfg = config();
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
%!		nn = nnInitNetwork(sizes,cfg);
%!		nn.weights;
%!		assert(numel(nn.weights)>0,'No element for weights!')
%!		assert(numel(nn.weights)==np,'Invalid number of NN parameters %d != %d\n',numel(nn.weights),np);
%!	end

% ==> It should have the proper weights if we use sparse initialization:

%!test
%!	num=1;
%!	cfg = config();
%!	cfg.use_sparse_init = true;
%!	for i=1:num,
%!		nl = random2(3,5);
%!		sizes = zeros(nl,1);
%!		si = random2(3,5);
%!		cfg.sparse_init_lot_size = si;
%!		for j=1:nl,
%!			sizes(j) = random2(8,10);
%!		end
%!		% compute the expected number of params:
%!		np = 0;
%!		nv = 0; % number of valid elements (eg. non zeros)
%!		for i=1:nl-1,
%!			np += (sizes(i)+1)*sizes(i+1);
%!			nv += sizes(i+1)*si; % si valid element per line in that matrix.
%!		end
%!	
%!		nn = nnInitNetwork(sizes,cfg);
%!		nn.weights;
%!		assert(numel(nn.weights)==np,'Invalid number of NN parameters %d != %d\n',numel(nn.weights),np);
%!		% Check the number of 0 elements:
%!		vec = nn.weights(:)!=0.0;
%!		len = sum(vec);
%!		assert(len==nv,'Invalid number of non zero elements: %d!=%d',len,nv)
%!	end
