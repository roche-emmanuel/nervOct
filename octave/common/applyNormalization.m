function [X_norm] = applyNormalization(X,mu,sigma)
% Apply normalization to a given matrix.

if isempty(X),
	X_norm = [];
	return;
end

assert(size(X,2)==size(mu,2),'Invalid size of mu');
assert(size(X,2)==size(sigma,2),'Invalid size of sigma');

n = size(X,1);

% warning("off", "Octave:broadcast");
X_norm = (X - repmat(mu,n,1)) ./ repmat(sigma,n,1);
% warning("on", "Octave:broadcast");

end

% ==> Ensure that normalisation can be applyed:
%!test
%!	X = rand(100,34)*10.0;
%!	mu = mean(X);
%!	sigma = std(X);
%!	Xn = applyNormalization(X,mu,sigma);
%!	
%!	% Since the mean is close to 0, the minimal value should be negative:
%!	mini = min(min(Xn));
%!	maxi = max(max(Xn));
%!	assert(mini<0,'Minimal value should be negative')
%!	assert(maxi>0,'Maximal value should be positive')
%!	
%!	% since we have an N(0,1) distribution, there should be practically no value outside of the range [-4sigma, 4 sigma]
%!	assert(sum(abs(mean(Xn)))<1e-9,'Mean value is out of range')
%!	assert(sum(abs(std(Xn)-1.0))<1e-9,'Sigma is out of range')

% ==> Should return an empty matrix if the input is empty:
%!test
%!	X = [];
%!	Xn = applyNormalization(X,1:4,1:4);
%!	assert(isempty(Xn),'Expecting result to be empty matrix')
