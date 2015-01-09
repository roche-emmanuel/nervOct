function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.


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

% Now get the max index per column:
[dum pred] = max(hx);


% ---------------------------------------------------------------------

end

