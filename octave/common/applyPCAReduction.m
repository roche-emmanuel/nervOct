function [X_red] = applyPCAReduction(X,reduce)
% Apply PCA reduction to a given matrix.

if isempty(X),
	X_red = [];
	return;
end

assert(size(X,2)==size(reduce,1),'Invalid size of reduce');
X_red = X * reduce;
end

% ==> Should return an empty matrix if the input is empty:
%!test
%!	X = [];
%!	Xn = applyPCAReduction(X,1:4);
%!	assert(isempty(Xn),'Expecting result to be empty matrix')
