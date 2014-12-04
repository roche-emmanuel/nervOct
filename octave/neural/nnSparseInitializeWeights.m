function W = nnSparseInitializeWeights(L_in, L_out, numN)

% Prepare a matrix of zeros.
W = zeros(L_out,1+L_in);

% Now for each line in this matrix (eg. for each neuron in this layer)
% We need to select numN input elements (not taking the bias into account)
if numN>L_in,
	error('number of selected active neurons must be less that total number of neurons.');
end

% Prepare the standard distribution values:
vals = stdnormal_rnd(L_out, numN);

for i=1:L_out,

	% we select the active neuron indices:
	% We apply an offset of 1 so that the bias term is never taken into account.
	idx = (randperm(L_in)(1:numN))+1;

	% Now we apply the gaussian values in the weight matrix:
	W(i,idx) = vals(i,:);
end

end

% ==> The resulting matrix should have the proper dimensions:
%!test
%!	W = nnSparseInitializeWeights(3,5,1);
%!	assert(size(W)==[5,4]);

% ==> For a given size, 2 matrices should not be the same:
%!test
%!	W1 = nnSparseInitializeWeights(3,5,1);
%!	W2 = nnSparseInitializeWeights(3,5,1);
%!	len = sum(sum(abs(W1-W2)));
%!	assert(len>0,'Matrices are the same ?')

% ==> For a given size, only a fixed number of elements should not be 0:
%!test
%!	W = nnSparseInitializeWeights(30,50,10);
%!	% Predicted number of zeros:
%!	pred = 50*31 - 50*10;
%!	vec = W(:)==0.0;
%!	len = sum(vec);
%!	assert(len==pred,'Invalid number of zero elements: %d!=%d',len,pred)
