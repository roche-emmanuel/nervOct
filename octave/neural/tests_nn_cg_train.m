
% The prototype to call the function should be:
% [params] = nn_cg_train(layer_sizes, X, yy, init_params, lambda, maxiter)

% ==> Should throw an error if invalid count of arguments:
%!error <nn_cg_train: Invalid number of arguments: 1> nn_cg_train(1)

% ==> Should throw an error if invalid type of arguments:
%!error <nn_cg_train: layer_sizes \(arg 0\) should be a matrix type> nn_cg_train(1,2,3,4,5,6)
%!error <nn_cg_train: X \(arg 1\) should be a matrix type> nn_cg_train(rand(2,2),2,3,4,5,6)
%!error <nn_cg_train: yy \(arg 2\) should be a matrix type> nn_cg_train(rand(2,2),[3 2 1],3,4,5,6)
%!error <nn_cg_train: init_params \(arg 3\) should be a matrix type> nn_cg_train([1 2 3],[3 2 1],rand(3,3),4,5,6)
%!error <nn_cg_train: lambda \(arg 4\) should be a double> nn_cg_train(rand(2,2),[3 2 1],rand(3,3),rand(4,5),false,6)
%!error <nn_cg_train: maxiter \(arg 5\) should be a double> nn_cg_train(rand(2,2),[3 2 1],rand(3,3),rand(4,5),0.1,false)

%!test nn_cg_train([3 2 1], rand(10,3),rand(10,2),rand(11,1),0.1,3);

% ==> Should produce the same results as the fmincg method:

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

%!test
%!	num=100
%!	for i=1:num,
%!		num_labels = 3; %random2(1,5);
%!		niter=5;
%!		nl = random2(3,5);
%!		lsizes=zeros(nl,1);
%!		for j=1:nl
%!			lsizes(j) = random2(3,6);
%!		end
%!		lsizes(end) = num_labels;
%!		%lsizes	
%!		m = random2(50,100);
%!		X  = nnDebugInitializeWeights(lsizes(1) - 1, m);
%!		y  = mod(1:m, num_labels)'; % We use 0-bazed labels.
%!		lambda = random2(1,10)/10;
%!	
%!		% Prepare the matrix of labels here:
%!		yy = nnBuildLabelMatrix(y);
%!	
%!		% Prepare the nn params:
%!		nl = size(lsizes,1);
%!	
%!		nn_params = [];
%!		cpu_params = [];
%!	
%!		nt = nl-1;
%!		
%!		% Retrieve the max iteration number from the training structure
%!		options = optimset('MaxIter', niter);
%!
%!		% We generate some 'random' test data
%!		for i=1:nt,
%!			theta = nnDebugInitializeWeights(lsizes(i), lsizes(i+1));
%!			nn_params = [nn_params; theta(:)];
%!			cpu_params = [cpu_params; theta(:)];
%!		end
%!	
%!		% Compare the cost values we can get out of the functions:
%!		[j1 grad1] = nnCostFunction(nn_params, lsizes, X, yy', lambda);
%!		[j2 grad2] = nn_cost_function_cuda(nn_params, lsizes, X, yy', lambda);	
%!
%!		ng = numel(grad1);
%!		assert(ng==numel(grad2),'Mismatch in number of gradients: %d!=%d',ng,numel(grad2))
%!		for j=1:ng,
%!			assert(abs(grad1(j)-grad2(j))<1e-10,'Mismatch at gradient %d: %.16f!=%.16f',j,grad1(j),grad2(j))
%!		end
%!
%!		% size(X)
%!		% size(yy)
%!		costFunc = @(p) nnCostFunction(p, lsizes, X, yy', lambda);
%!
%!		nn_params = fmincg(costFunc, nn_params, options);
%!		cpu_params = nn_cg_train(lsizes', X, yy', cpu_params, lambda, niter);
%!	
%!		% Compare the cpu params and the octave params:
%!		np = numel(nn_params);
%!		assert(np==numel(cpu_params),'Mismatch in number of parameters: %d!=%d',np,numel(cpu_params))
%!	
%!		for j=1:np,
%!			assert(abs(nn_params(j)-cpu_params(j))<1e-8,'Mismatch at parameter %d: %.16f!=%.16f',j,nn_params(j),cpu_params(j))
%!		end
%!	end

% ==> Check the performances for a not too small network:
%!test
%!	num_labels = 3;
%!	lsizes = [1441; 200; num_labels];
%!	m = 2000;
%!	X  = nnDebugInitializeWeights(lsizes(1) - 1, m);
%!	y  = mod(1:m, num_labels)'; % We use 0-bazed labels.
%!	lambda = 0.1;
%!	
%!	% Prepare the matrix of labels here:
%!	yy = zeros(num_labels,m);
%!	
%!	for c=1:m,
%!		% Note that the labels are 0-based, so we need to offset them by one:
%!		yy(y(c)+1,c)=1;
%!	end;
%!	
%!	% Prepare the nn params:
%!	nl = size(lsizes,1);
%!	
%!	nn_params = [];
%!	nt = nl-1;
%!	
%!	% We generate some 'random' test data
%!	for i=1:nt,
%!		theta = nnDebugInitializeWeights(lsizes(i), lsizes(i+1));
%!		nn_params = [nn_params; theta(:)];
%!	end
%!	size(X)
%!	size(yy)
%!	
%!	niter=10;
%!	profile on;
%!	tic()
%!	nn_params = nn_cg_train(lsizes, X, yy, nn_params, lambda, niter);
%! 	toc()
%!	profile off;
%!	profshow(profile('info'))

% ==> Should be able to train proper dimension network

%test
%	cfg = config();
%	tr = nnPrepareTraining(1:1,cfg);
%	nn = nnInitNetwork([tr.num_features 10 3],cfg);
%	X = tr.X_train;
%	y = tr.y_train;
%	
%	% Before continuing we have to select the target symbol we want to train on.
%	y = y(:,cfg.target_symbol_pair);
%	
%	% Now we need to convert the label vector into a matrix:
%	yy = nnBuildLabelMatrix(y)';
%	
%	lambda = 0.1;
%	niter=50;
%	lsizes=nn.layer_sizes
%	weights=nn.weights;
%	lsizes
%	size(X)
%	size(yy)
%	% fprintf('Starting test...\n')
%	nn_params = nn_cg_train(lsizes, X, yy, weights, lambda, niter);
%	% fprintf('Stopping test...\n')
