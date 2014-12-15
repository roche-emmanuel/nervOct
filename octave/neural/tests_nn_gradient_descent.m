
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


% ==> Should throw if lsizes is not a matrix:

%!function no_lsizes_mat()
%!  desc.lsizes = 10;
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: lsizes is not a matrix type> no_lsizes_mat()


% ==> Should throw an error if no X_train is provided in input structure:

%!function no_X_train()
%!  desc.lsizes = [4,3,2];
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: X_train value is not defined> no_X_train()


% ==> Should throw an error if X_train is not a matrix:

%!function no_X_train_mat()
%!  desc.lsizes = [4,3,2];
%!  desc.X_train = 10;
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: X_train is not a matrix type> no_X_train_mat()


% ==> Should throw an error if no params is provided in input structure:

%!function no_params()
%!  desc.lsizes = [4,3,2];
%!  desc.X_train = rand(10,4);
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: params value is not defined> no_params()


% ==> Should throw an error if params is not a matrix:

%!function no_params_mat()
%!  desc.lsizes = [4,3,2];
%!  desc.X_train = rand(10,4);
%!  desc.params = 10;
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: params is not a matrix type> no_params_mat()


% ==> Should throw an error if no y_train is provided in input structure:

%!function no_y_train()
%!  desc.lsizes = [4,3,2];
%!  desc.X_train = rand(10,4);
%!  desc.params = rand(10,1);
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: y_train value is not defined> no_y_train()


% ==> Should throw an error if y_train is not a matrix:

%!function no_y_train_mat()
%!  desc.lsizes = [4,3,2];
%!  desc.X_train = rand(10,4);
%!  desc.params = rand(10,1);
%!  desc.y_train = 10;
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: y_train is not a matrix type> no_y_train_mat()


%%%% advanced checks: 

% ==> Should throw an error if X_train ncols doesn't match the lsizes(1) value:

%!function no_X_train_match()
%!  desc.lsizes = [4,3,2];
%!  desc.X_train = rand(10,5);
%!  desc.y_train = rand(10,2);
%!  desc.params = rand(10,1);
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: Feature matrix doesn't match lsizes: 5!=4> no_X_train_match()


% ==> Should throw an error if params size doesn't match expected size from lsizes:

%!function no_params_match()
%!  desc.lsizes = [4,3,2];
%!  desc.X_train = rand(10,4);
%!  desc.y_train = rand(10,2);
%!  desc.params = rand(10,1);
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: params doesn't match expected size: 10!=23> no_params_match()


% ==> Should throw an error if we don't have the same number of samples according to
% X_train and y_train:

%!function no_nsamples_match()
%!  desc.lsizes = [4,3,2];
%!  desc.X_train = rand(10,4);
%!  desc.y_train = rand(9,2);
%!  desc.params = rand(23,1);
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: mismatch in nsamples_train: 10!=9> no_nsamples_match()


% ==> Should throw an error if y_trian doesn't match the lsizes specs:

%!function no_y_train_match()
%!  desc.lsizes = [4,5,2];
%!  desc.X_train = rand(10,4);
%!  desc.y_train = rand(10,3);
%!  desc.params = rand(37,1);
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: y_train doesn't match lsizes: 3!=2> no_y_train_match()

% ==> Should throw an error if momentum is not of the proper type:

%!function invalid_momentum()
%!  desc.lsizes = [4,5,2];
%!  desc.X_train = rand(10,4);
%!  desc.y_train = rand(10,2);
%!  desc.params = rand(37,1);
%!  desc.momentum = true;
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: momentum is not a double type> invalid_momentum()

% ==> Should throw an error if momentum is out of range

%!function invalid_momentum_val()
%!  desc.lsizes = [4,5,2];
%!  desc.X_train = rand(10,4);
%!  desc.y_train = rand(10,2);
%!  desc.params = rand(37,1);
%!  desc.momentum = 1.1;
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: invalid value for momentum: 1.1> invalid_momentum_val()


% ==> Should throw an error if epsilon is not of the proper type:

%!function invalid_epsilon()
%!  desc.lsizes = [4,5,2];
%!  desc.X_train = rand(10,4);
%!  desc.y_train = rand(10,2);
%!  desc.params = rand(37,1);
%!  desc.momentum = 0.9;
%!  desc.epsilon = true;
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: epsilon is not a double type> invalid_epsilon()


% ==> Should throw an error if epsilon is out of range

%!function invalid_epsilon_val()
%!  desc.lsizes = [4,5,2];
%!  desc.X_train = rand(10,4);
%!  desc.y_train = rand(10,2);
%!  desc.params = rand(37,1);
%!  desc.momentum = 0.9;
%!  desc.epsilon = 0;
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: invalid value for epsilon: 0> invalid_epsilon_val()


% ==> Should throw an error if dropout size is incorrect:

%!function invalid_dropouts()
%!  desc.lsizes = [4,5,2];
%!  desc.X_train = rand(10,4);
%!  desc.y_train = rand(10,2);
%!  desc.params = rand(37,1);
%!  desc.momentum = 0.9;
%!  desc.epsilon = 0.005;
%!  desc.dropouts = [0.8,0.5,0.5];
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: invalid size for dropout matrix size: 3!=2> invalid_dropouts()

% ==> Should throw an error if dropout is out of range:

%!function invalid_dropouts_val()
%!  desc.lsizes = [4,5,2];
%!  desc.X_train = rand(10,4);
%!  desc.y_train = rand(10,2);
%!  desc.params = rand(37,1);
%!  desc.momentum = 0.9;
%!  desc.epsilon = 0.005;
%!  desc.dropouts = [0.8,1.5];
%!	nn_gradient_descent(desc)
%!endfunction

%!error <nn_gradient_descent: dropout for layer 1 is out of range> invalid_dropouts_val()


% ==> Should work if the input argument is a structure:
%!test
%!	desc.lsizes = [3, 4, 1];
%!	desc.X_train = rand(100,3);
%!	desc.y_train = rand(100,1);
%!  desc.params = rand(21,1);
%!  desc.epsilon = 0.05;
%!	nn_gradient_descent(desc);

