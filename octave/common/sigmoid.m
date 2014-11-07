function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
% g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


g = 1 ./ (1+exp(-z));

%!function g = slow_sigmoid(z)
%!  v = exp(-z) + 1;
%!  g = 1 ./ v;
%!endfunction

%!test
%!	num = 100;
%!	for i=1:num
%!		sca = (rand(1,1)-0.5)*10.0;
%!		assert(sigmoid(sca)==slow_sigmoid(sca))
%!		vec = (rand(10,1)-0.5)*10.0;
%!		assert(sigmoid(vec)==slow_sigmoid(vec))
%!		m = (rand(10,10)-0.5)*10.0;
%!		assert(sigmoid(m)==slow_sigmoid(m))
%!	end
