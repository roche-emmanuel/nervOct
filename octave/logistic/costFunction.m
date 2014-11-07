function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
% m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

z = X*theta; % compute the thetaT*x for each example.
hx = sigmoid(z); %Take the sigmoid to build the actual prediction value.


m = size(X,1); % number of rows in X.

J = -(y .* log(hx) + (1-y) .* log(1 - hx));

% finally perform the sumation:
J = sum(J)/m;

% Now compute the gradient vector:

num = size(X,2);

pred = hx-y;
% W = zeros(m,num);

% for i=1:num,
%     W(:,i) = X(:,i) .* pred; 
% end
W = X .* pred;

% Then we sum the result:
delta = sum(W)/m; %this gives us a 1 x num matrix.

% grad should be a num x 1 matrix instead, so, we transpose:
grad = delta';



% =============================================================

end
