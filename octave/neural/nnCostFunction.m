function [J grad] = nnCostFunction(nn_params, layer_sizes, X, yy, lambda)

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Check how may layers we have:
nl = numel(layer_sizes);

% when we have nl layers, we have nl-1 weights matrices.
% That we should rebuild here:
nt = nl-1;

% Reshape nn_params back into the parameters Thetas i, the weight matrices for our neural network:
Thetas = cell(1,nt);
Activation = cell(1,nl);
Deltas = cell(1,nl);
Inputs = cell(1,nl);

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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1:
% First we need to compute h(x) for each example.

a = X';
Activation{1,1} = [ones(1,m); a];

for i=1:nt,
	% First we add the intercept term as a top row on the current activation:
	a = Activation{1,i};

	% size(Thetas{1,i})
	% size(a)

	% Then we can compute the total input for the next layer:
	z = Thetas{1,i} * a;

	% We have to store the z input for the backpropagation later:
	Inputs{1,i+1} = z;

	% Compute the output of the next layer:
	a = sigmoid(z);

	% Also save the activation value:
	Activation{1,i+1} = [ones(1,m); a];
end


% Now the final result is stored in a, the activation vector (eg. output) of the last layer.
% Thus we can rename this output as hx:
hx = a;

if any(hx>=1.0)
	error('Invalid weights or features, we should never have hx>=1.0')
end
if any(hx<=0.0)
	error('Invalid weights or features, we should never have hx<=0.0')
end

% Now we need to to compute the cost function:

% Now we build the cost "matrix":
cmat = yy .* log(hx) + (1-yy) .* log(1-hx);

% Then we perform the summation on all classes and on all examples:
J=-sum(sum(cmat))/m;

% Now we add the regularization terms:
b=lambda/(2*m);

reg = 0;
for i=1:nt,
	smat = Thetas{1,i}(:,2:end);
	reg += sum(sum(smat .* smat));
end

J = J + b * reg;


% Part 2: 
% now implementing back propagation.

% We start with the latest delta value:
delta = hx - yy;
Deltas{1,nl} = delta;

for i=nt:-1:2,
	% We have to accumulate the correction for each sample
	delta = (Thetas{1,i})' * delta;

	% Then we remove the first row:
	delta = delta(2:end,:);

	% And we multiply by the sigmoid gradient of the previous activation value:
	z = Inputs{1,i};

	delta = delta .* sigmoidGradient(z);

	% Save that delta value:
	Deltas{1,i} = delta;
end

% So now delta2 is a matrix of vector columns, so is a1
% delta3 is a matrix of vector columns, so is a2
% what we need to do is to extract each column from those matrices and multiply them to get a delta matrix, this matrix is then accumulated on the gradient matrices:

% Now we can compute the theta grad values:
np = size(nn_params,1);
grad = zeros(np,1);
pos = 1;

for i=1:nt,
	% Also add the regularization term at the same time:
	Theta = Thetas{1,i};
	n1 = size(Theta,1);
	n2 = size(Theta,2);

	reg = [zeros(n1,1) Theta(:,2:end)];

	delta = Deltas{1,i+1};
	act = Activation{1,i}';
	A = ones(size(delta,1),size(delta,2));
	B = ones(size(act,1),size(act,2));

	%mat = (Deltas{1,i+1} * Activation{1,i}' + lambda * reg)/m; 
	if i==1,
		% act = [ones(size(B,1),1) zeros(size(B,1),size(B,2)-1)];
		mat = (delta * act)/m; % + lambda * reg)/m;
		% mat
		% n1
		% n2
		% assert(mat(1,2)==0,'Invalid value for mat(1,2)=%f',mat(1,2)) 
	else
		mat = (delta * act)/m; % + lambda * reg)/m; 
	end

	% mat = (Deltas{1,i+1} * Activation{1,i}')/m;
	
	% mat = lambda * reg/m;
	% mat = ones(n1,n2)*lambda;

	count = n1*n2;
	grad(pos:pos+count-1,1) = mat(:)';
	pos += count;
end

% grad = Activation{1,nt};
% grad = Deltas{1,nt+1};

end

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

% ==> Check the values of the gradients:
%!test
%!	num=5;
%!	for i=1:num,
%!		nl = random2(3,5);
%!		sizes = zeros(nl,1);
%!		for j=1:nl,
%!			sizes(j) = random2(3,6);
%!		end
%!		[numgrad, grad] = nnCheckGradients(sizes,random2(5,10),rand(1,1));
%!		diff = norm(numgrad-grad)/norm(numgrad+grad);
%!		if(diff>1e-9),
%!			disp([numgrad grad]);
%!			error('Mismatch in numerical and computed gradients')
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
%!	costFunc = @(p) nnCostFunction(p, lsizes, X, yy, lambda);
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
