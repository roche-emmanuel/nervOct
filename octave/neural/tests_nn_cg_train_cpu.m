
% The prototype to call the function should be:
% [params] = nn_cg_train_cpu(layer_sizes, X, yy, init_params, lambda, maxiter)

% ==> Should throw an error if invalid count of arguments:
%!error <nn_cg_train_cpu: Invalid number of arguments: 1> nn_cg_train_cpu(1)

% ==> Should throw an error if invalid type of arguments:
%!error <nn_cg_train_cpu: layer_sizes \(arg 0\) should be a matrix type> nn_cg_train_cpu(1,2,3,4,5,6)
%!error <nn_cg_train_cpu: X \(arg 1\) should be a matrix type> nn_cg_train_cpu(rand(2,2),2,3,4,5,6)
%!error <nn_cg_train_cpu: yy \(arg 2\) should be a matrix type> nn_cg_train_cpu(rand(2,2),[3 2 1],3,4,5,6)
%!error <nn_cg_train_cpu: init_params \(arg 3\) should be a matrix type> nn_cg_train_cpu([1 2 3],[3 2 1],rand(3,3),4,5,6)
%!error <nn_cg_train_cpu: lambda \(arg 4\) should be a double> nn_cg_train_cpu(rand(2,2),[3 2 1],rand(3,3),rand(4,5),false,6)
%!error <nn_cg_train_cpu: maxiter \(arg 5\) should be a double> nn_cg_train_cpu(rand(2,2),[3 2 1],rand(3,3),rand(4,5),0.1,false)

%!test nn_cg_train_cpu([3 2 1], rand(10,3),rand(10,2),rand(11,1),0.1,3);

% ==> Should produce the same results as the fmincg method:

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

%!test
%!	num=10
%!	for i=1:num,
%!		num_labels = 3; %random2(1,5);
%!		niter=1
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
%!		% size(X)
%!		% size(yy)
%!		costFunc = @(p) nnCostFunction(p, lsizes, X, yy', lambda);
%!		% costFuncCPU = @(p) nn_cost_function_cuda(p, lsizes, X, yy, lambda);	
%!
%!		%for j=1:niter,
%!			nn_params = fmincg(costFunc, nn_params, options);
%!			cpu_params = fmincg(costFunc, cpu_params, options);
%!		%end
%!	
%!		% Compare the cpu params and the octave params:
%!		np = numel(nn_params);
%!		assert(np==numel(cpu_params),'Mismatch in number of parameters: %d!=%d',np,numel(cpu_params))
%!	
%!		for j=1:np,
%!			assert(abs(nn_params(j)-cpu_params(j))<1e-10,'Mismatch at parameter %d: %f!=%f',j,nn_params(j),cpu_params(j))
%!		end
%!	end
