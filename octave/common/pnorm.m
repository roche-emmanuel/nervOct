% Compute the pnorm of a given vector
% or for each column in a matrix:
function res = pnorm(inputs,p)
	res = sum(abs(inputs) .^ p) .^ (1/p);
end
