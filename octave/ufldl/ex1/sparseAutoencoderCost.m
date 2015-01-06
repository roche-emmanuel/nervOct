function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

warning ("error", "Octave:broadcast");

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% First we should compute the cost:

% number of samples:
ns = size(data,2);

% perform forward pass:
b1 = repmat(b1,1,ns);
b2 = repmat(b2,1,ns);

z2 = W1*data + b1;
a2 = sigmoid(z2);
z3 = W2*a2 + b2;
hx = sigmoid(z3);

% Now we can compute the first part of the cost which is the difference between hx and the data itself:
dif = hx-data;
dif = dif .* dif;

cost = sum(sum(dif))/(2*ns);

% Then we should had the weight decay:
cost = cost + (sum(sum(W1 .* W1)) + sum(sum(W2 .* W2)))*lambda/2.0;

% Finally we should add here the sparsity penalty:
% To do so we first need to compute the mean activation:
% To perform this operation we can simply sum on each column of a2 and divide by ns:
rho = sum(a2,2)/ns;

KL = sparsityParam * log(sparsityParam ./ rho) + (1.0 - sparsityParam) * log( (1.0 - sparsityParam) ./ (1.0 - rho) );

Jsp = beta * sum(KL);
% fprintf('Computed sparsity penalty: %f.\n',Jsp)
cost = cost + Jsp;

% Now we need to compute the gradients:

% First we need to compute the deltas:
d3 = - (data-hx) .* (hx .* (1.0 - hx)); 

% We can apply the sparsity penalty here:
sp = beta * ( (-sparsityParam ./ rho) + ((1.0 - sparsityParam) ./ (1.0 - rho)) );

% We need to repeat this matrix sp by the number of samples:
sp = repmat(sp,1,ns);

d2 = (W2' * d3 + sp)  .* (a2 .* (1.0 - a2));

W1grad = (d2 * data')/ns + lambda * W1;
b1grad = sum(d2,2)/ns;

W2grad = (d3 * a2')/ns + lambda * W2;
b2grad = sum(d3,2)/ns;

% size(W1grad)
% size(b1grad)
% size(W2grad)
% size(b2grad)

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
