function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

ns = size(data, 2);

groundTruth = full(sparse(labels, 1:ns, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% First step is to compute the hx value:
% data contains one sample per column.
% theta is a matrix of size numClasses * inputSize
% Note that the inputSize takes the intercept term into account.

% compute the core value of hx:
hx = theta * data;

% To avoid overflow we compute the max on each column and we substract it:
hx = bsxfun(@minus, hx, max(hx, [], 1));
% hx = hx - repmat(max(hx),numClasses,1);

% Then we can take the exponential:
hx = exp(hx);

% Then we compute the sumation for each column:
% denom = repmat(sum(hx),numClasses,1);

% Then we renormalize the hx value with this denominator value:
% hx = hx ./ denom;
hx = bsxfun(@rdivide, hx, sum(hx));

% Now we can compute the cost:
cost = - sum(sum(groundTruth .* log(hx)))/ns;

% Add the weight decay part:
cost += sum(sum(theta(:,2:end) .* theta(:,2:end))) * lambda / 2.0;

% now we should compute the gradients:

% We need the following matrix to have numClasses colunms and ns rows:
M = (groundTruth - hx);

% Then we can compute the grads for all thetas at the same time:
thetagrad = - (M * data') / ns + lambda * [zeros(size(theta,1),1) theta(:,2:end)];

% ------------------------------------------------------------------l
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

