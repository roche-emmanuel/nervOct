
% The prototype to call the function should be:
% [id] = nn_create_traits(desc)
% [] = nn_destroy_traits(id)


% ==> It should be possible to create/destroy simple traits

%!function theta = initializeParameters(hiddenSize, visibleSize)
%!
%!	%% Initialize parameters randomly based on layer sizes.
%!	r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
%!	W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
%!	W2 = rand(visibleSize, hiddenSize) * 2 * r - r;
%!	
%!	b1 = zeros(hiddenSize, 1);
%!	b2 = zeros(visibleSize, 1);
%!	
%!	% Convert weights and bias gradients to the vector form.
%!	% This step will "unroll" (flatten and concatenate together) all 
%!	% your parameters into a vector, which can then be used with minFunc. 
%!	theta = [b1(:) ; W1(:) ; b2(:) ; W2(:)];
%!
%!endfunction

%!test
%!	visibleSize = 64;
%!	hiddenSize = 25;
%!	patches = rand(64,10000);
%!	
%!	desc.lsizes = [visibleSize hiddenSize visibleSize];
%!	desc.X_train = patches;
%!	desc.y_train = patches;
%!	
%!	desc.params = initializeParameters(hiddenSize, visibleSize);
%!	
%!	desc.lambda = 0.0001;
%!	desc.useSoftmax = false;
%!	desc.spaeBeta = 3;
%!	desc.spaeSparsity = 0.01;
%!	
%!	id = nn_create_traits(desc);
%!	assert(id>=1,'Invalid ID value: %d',id);
%!	% We should now be able to compute the cost with those parameters:
%!	desc.id = id;
%!	[cost grads] = nn_costfunc_device(desc);
%!	
%!	fprintf('Cost value is: %f\n',cost);
%!	nn_destroy_traits(id);
