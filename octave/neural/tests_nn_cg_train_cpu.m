
% The prototype to call the function should be:
% [params] = nn_cg_train_cpu(layer_sizes, X, yy, init_params, lambda, maxiter)

% ==> Should throw an error if invalid count of arguments:
%!error <nn_cg_train_cpu: Invalid number of arguments: 1> nn_cg_train_cpu(1)

% ==> Should throw an error if invalid type of arguments:
%!error <nn_cg_train_cpu: layer_sizes \(arg 0\) should be a matrix type> nn_cg_train_cpu(1,2,3,4,5,6)
%!error <nn_cg_train_cpu: X \(arg 1\) should be a matrix type> nn_cg_train_cpu(rand(2,2),2,3,4,5,6)
%!error <nn_cg_train_cpu: yy \(arg 2\) should be a matrix type> nn_cg_train_cpu(rand(2,2),[3 2 1],3,4,5,6)
%!error <nn_cg_train_cpu: init_params \(arg 3\) should be a matrix type> nn_cg_train_cpu([1 2 3],[3 2 1],rand(3,3),4,5,6)
%!error <nn_cg_train_cpu: lambda \(arg 4\) should be a double> nn_cg_train_cpu(rand(2,2),[3 2 1],rand(3,3),rand(4,5),false,6)
%!error <nn_cg_train_cpu: maxiter \(arg 5\) should be a double> nn_cg_train_cpu(rand(2,2),[3 2 1],rand(3,3),rand(4,5),0.1,false)

%!test nn_cg_train_cpu([3 2 1], rand(10,3),rand(10,2),rand(11,1),0.1,3)

% ==> Should produce the same results as the nnCostFunction method:

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction
