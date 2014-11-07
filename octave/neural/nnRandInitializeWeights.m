function W = nnRandInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the column row of W handles the "bias" terms
%

%
% Note: The first row of W corresponds to the parameters for the bias units
%

eps=sqrt(6)/sqrt(L_in+L_out);

W = rand(L_out, 1+L_in)*2*eps - eps;

end

% ==> The resulting matrix should have the proper dimensions:
%!test
%!	W = nnRandInitializeWeights(3,5);
%!	assert(size(W)==[5,4]);

% ==> It should output values in the expected range:
%!test
%! 	W = nnRandInitializeWeights(3,5);
%!	maxi = max(max(W));
%!	mini = min(min(W));
%!	eps = sqrt(6)/sqrt(3+5);
%!	assert(maxi <= eps)
%!	assert(mini >= -eps)

% ==> For a given size, 2 matrices should not be the same:
%!test
%!	W1 = nnRandInitializeWeights(3,5);
%!	W2 = nnRandInitializeWeights(3,5);
%!	len = sum(sum(abs(W1-W2)));
%!	assert(len>0,'Matrices are the same ?')
