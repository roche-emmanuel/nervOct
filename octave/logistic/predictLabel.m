function [p, mval] = predictLabel(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

% Theta should be a matrix with num_labels columns,
% So we compute the predicted value for each column
[mval, p] = max(sigmoid(X*theta),[],2);

% =========================================================================


end
