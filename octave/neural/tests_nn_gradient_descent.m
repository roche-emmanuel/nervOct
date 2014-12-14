
% The prototype to call the function should be:
% [results] = nn_gradient_descent(struct)
% 

% ==> Should throw an error if no argument is provided:
%!error <nn_gradient_descent: Invalid number of arguments: 0> nn_gradient_descent()
%!error <nn_gradient_descent: Invalid number of arguments: 2> nn_gradient_descent(1,2)

% ==> Should throw an error if no input structure is provided:
%!error <nn_gradient_descent: desc \(arg 0\) should be a structure type> nn_gradient_descent(1)

% ==> Should throw an error if no lsizes is provided in input structure:

%!function no_lsizes()
%!  desc.dummy = 10;
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: lsizes value is not defined> no_lsizes()

% ==> Should work if the input argument is a structure:
%!test
%!	desc.lsizes = [3, 4, 1];
%!	nn_gradient_descent(desc);

