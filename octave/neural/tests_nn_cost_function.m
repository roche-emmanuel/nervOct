
% The prototype to call the function should be:
% [J grad] = nn_cost_function(nn_params, layer_sizes, X, yy, lambda)

% ==> Should throw an error if invalid count of arguments:
%!error <nn_cost_function: Invalid number of arguments: 1> nn_cost_function(1)

% ==> Should throw an error if invalid type of arguments:
%!error <nn_cost_function: nn_params \(arg 0\) should be a matrix type> nn_cost_function(1,2,3,4,5)
%!error <nn_cost_function: layer_sizes \(arg 1\) should be a matrix type> nn_cost_function(rand(2,2),2,3,4,5)
%!error <nn_cost_function: X \(arg 2\) should be a matrix type> nn_cost_function(rand(2,2),[3 2 1],3,4,5)
%!error <nn_cost_function: yy \(arg 3\) should be a matrix type> nn_cost_function(rand(2,2),[3 2 1],rand(3,3),4,5)
%!error <nn_cost_function: lambda \(arg 4\) should be a double> nn_cost_function(rand(2,2),[3 2 1],rand(3,3),rand(4,5),false)

% ==> Should produce the same results as the nnCostFunction method:

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

%!test
%!	num=100;
%!	for i=1:num,
%!		num_labels = random2(1,5);
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
%!		yy = zeros(num_labels,m);
%!	
%!		for c=1:m,
%!			% Note that the labels are 0-based, so we need to offset them by one:
%!			yy(y(c)+1,c)=1;
%!		end;
%!	
%!		% Prepare the nn params:
%!		nl = size(lsizes,1);
%!	
%!		nn_params = [];
%!		nt = nl-1;
%!	
%!		% We generate some 'random' test data
%!		for i=1:nt,
%!			theta = nnDebugInitializeWeights(lsizes(i), lsizes(i+1));
%!			nn_params = [nn_params; theta(:)];
%!		end
%!		% size(X)
%!		% size(yy)
%!		costFunc = @(p) nnCostFunction(p, lsizes, X, yy, lambda);
%!		costFuncCPU = @(p) nn_cost_function(p, lsizes, X, yy, lambda);
%!	
%!		% check that we have the same values:
%!		[j1 grad1] = costFunc(nn_params);
%!		[j2 grad2]= costFuncCPU(nn_params);
%!		%size(grad1)
%!		%size(grad2)
%!		%grad2 = grad1;
%!		assert(abs(j1-j2)<1e-10,'Mismatch in computed cost value: %.16f!=%.16f',j1,j2);
%!		len = sum(sum(abs(grad1 - grad2)));
%!		assert(len<1e-10,'Mismatch in computed gradients value: len=%.16f',len);
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
%!	costFunc = @(p) nn_cost_function(p, lsizes, X, yy, lambda);
%!	
%!	num=10;
%!	profile on;
%!	tic()
%!	for i=1:num,
%!		costFunc(nn_params);
%!	end
%! toc()
%!	profile off;
%!	profshow(profile('info'))
