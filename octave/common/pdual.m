% Retrieve the dual q of p,
% such as 1/p + 1/q = 1;
function q = pdual(p)
	q = 1.0/(1.0 - 1.0/p);
end
