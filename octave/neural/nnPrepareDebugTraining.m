function trdata = nnPrepareDebugTraining(num_samples, num_features, num_labels, cfg)
% Method used to prepare a dummy dataset with random values to train on.

% Prepare the matrix to hold the features:
m = num_samples;
count = m*num_features;

X = reshape(sin(1:count), m, num_features) / 10;

% prepare the matrix to hold the labels:
y = mod(1:m, num_labels)';

y = repmat(y,1,cfg.num_symbol_pairs);

% Now that we have the raw feature and label matrices, we have to separate them in train/cv/test sets using the set ratios 
% from the config:
ratios = cfg.dataset_ratios;
[X_train, X_cv, X_test] = splitDataset(X, ratios(1), ratios(2), ratios(3));
[y_train, y_cv, y_test] = splitDataset(y, ratios(1), ratios(2), ratios(3));

% Once the datasets are separated, we need to compute and apply feature normalization:
% The normalisation is computed fromt the train dataset only:
mu = mean(X_train);
sigma = std(X_train);

% Then we apply the normalization on all datasets:
X_train = applyNormalization(X_train,mu,sigma);
X_cv = applyNormalization(X_cv,mu,sigma);
X_test = applyNormalization(X_test,mu,sigma);

% Prepare the structure to return
trdata.X_train = X_train;
trdata.X_cv = X_cv;
trdata.X_test = X_test;
trdata.y_train = y_train;
trdata.y_cv = y_cv;
trdata.y_test = y_test;

% Also save the mu and sigma value for future normalizations:
trdata.mu = mu;
trdata.sigma = sigma;

% by default init the regularization parameter to 0:
trdata.regularization_param = cfg.default_regularization_parameter;

% by default the max iteration number is fixed by config:
trdata.max_iterations = cfg.default_max_training_iterations;

% proportion of the data to use for the training:
trdata.train_ratio = 1.0;

% deep training status:
trdata.deep_training = cfg.default_deep_training;

end

% ==> Should return a struct containing an Xtrain matrix and Ytrain vector, both with same number of rows:

%!test
%!	cfg = config();
%!	data = nnPrepareDebugTraining(100,5,3,cfg);
%!	n1 = size(data.X_train,1);
%!	n2 = size(data.y_train,1);
%!	assert(n1==n2,'Mismatch between X_train and y_train')
%!	n1 = size(data.X_cv,1);
%!	n2 = size(data.y_cv,1);
%!	assert(n1==n2,'Mismatch between X_cv and y_cv')
%!	n1 = size(data.mu,2);
%!	n2 = size(data.sigma,2);
%!	assert(n1==n2,'Mismatch between mu and sigma')
%!	assert(n1==5,'Invalid col count in mu and sigma')
%!  assert(size(data.y_train,2)==cfg.num_symbol_pairs,'Invalid number of label colunms')
%! 	% Also check that the normalization is applied properly:
%!	Xn = data.X_train;
%!	assert(sum(abs(mean(Xn)))<1e-9,'Mean value is out of range')
%!	assert(sum(abs(std(Xn)-1.0))<1e-9,'Sigma is out of range')
%!	assert(data.regularization_param==cfg.default_regularization_parameter,'Invalid regularization parameter value')
%!	assert(data.max_iterations==cfg.default_max_training_iterations,'Invalid max iterations value')
%!	assert(data.train_ratio==1.0,'Invalid train_ratio value')
%!	assert(data.deep_training==cfg.default_deep_training,'Invalid deep_training value')
