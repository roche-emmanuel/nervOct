function [numgrad grad] = nnCheckGradients(lsizes, nsamples = 5, lambda = 0)
%CHECKNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%

% assert(size(lsizes) == [3, 1],'Can only check 3 layers NN for now.')

num_labels = lsizes(end);
m = nsamples;

% Number of layers to consider for the test:
nl = size(lsizes,1);

nn_params = [];
nt = nl-1;

% We generate some 'random' test data
for i=1:nt,
	theta = nnDebugInitializeWeights(lsizes(i), lsizes(i+1));
	nn_params = [nn_params; theta(:)];
end

% Reusing nnDebugInitializeWeights to generate X
X  = nnDebugInitializeWeights(lsizes(1) - 1, m);
y  = mod(1:m, num_labels)'; % We use 0-bazed labels.


% Prepare the matrix of labels here:
yy = zeros(num_labels,m);

for c=1:m,
	% Note that the labels are 0-based, so we need to offset them by one:
	yy(y(c)+1,c)=1;
end;

% Short hand for cost function
costFunc = @(p) nnCostFunction(p, lsizes, X, yy, lambda);

[cost, grad] = costFunc(nn_params);
numgrad = computeNumericalGradient(costFunc, nn_params);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
% disp([numgrad grad]);
% fprintf(['The above two columns you get should be very similar.\n' ...
%          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% % Evaluate the norm of the difference between two solutions.  
% % If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% % in computeNumericalGradient.m, then diff below should be less than 1e-9
% diff = norm(numgrad-grad)/norm(numgrad+grad);

% fprintf(['If your backpropagation implementation is correct, then \n' ...
%          'the relative difference will be small (less than 1e-9). \n' ...
%          '\nRelative Difference: %g\n'], diff);

end
