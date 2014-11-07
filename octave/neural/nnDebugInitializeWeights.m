function W = nnDebugInitializeWeights(L_in, L_out)
%DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections using a fixed
%strategy, this will help you later in debugging
%   W = DEBUGINITIALIZEWEIGHTS(L_in, L_out) initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections using a fix set of values
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first row of W handles the "bias" terms
%

% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
W = reshape(sin(1:(L_out*(L_in+1))), L_out, L_in +1) / 10;

end

% ==> The resulting matrix should have the proper dimensions:
%!test
%!	W = nnDebugInitializeWeights(3,5);
%!	assert(size(W)==[5,4]);

% ==> It should output values in the expected range:
%!test
%! 	W = nnDebugInitializeWeights(3,5);
%!	maxi = max(max(W));
%!	mini = min(min(W));
%!	eps = 0.1; % This is the max range of sin(x)/10.0
%!	assert(maxi <= eps)
%!	assert(mini >= -eps)

% ==> For a given size, the matrix should alwas be the same:
%!test
%!	W1 = nnDebugInitializeWeights(3,5);
%!	W2 = nnDebugInitializeWeights(3,5);
%!	len = sum(sum(abs(W1-W2)));
%!	assert(len==0,'Matrices are not the same.')
