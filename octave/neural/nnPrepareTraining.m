function trdata = nnPrepareTraining(weeks,cfg)
% method used to prepare the training data for a given number of weeks,
% the weeks to be selected are passed as an input vector.
% This method will output a warning if one of the week requested is not available.

% The resulting output is a structure containing the following fields:
% trdata.X_train : the feature matrice to use for training
% trdata.y_train : the label vector to sue for training (same length as the Xtrain feature matrice)

assert(exist('weeks', 'var') == 1, 'must provide a week vector')

% Retrieve the number of weeks to load:
nweeks = numel(weeks);

% Prepare the pattern to use to load a file:
fpattern = [cfg.datapath '/' cfg.week_feature_pattern];

% Prepare the matrix to hold the features:
X = [];

% prepare the matrix to hold the labels:
y = [];

% Matrix used to hodl the prices:
prices = [];

% Number of rows to expect per week:
nrows = cfg.num_week_minutes - cfg.num_pred_bars - cfg.num_input_bars + 1;

fprintf('Loading weeks data...\n')

% Here we load the complete week dataset:
week_datasets = loadData(cfg.datapath,cfg.week_dataset_name,'week_data');

% Retrieve the total number of weeks:
total_nweeks = size(week_datasets,2);

assert(max(nweeks)<=total_nweeks,'Invalid access to out of range week: %d>%d',max(nweeks),total_nweeks)

for i=1:nweeks
	wid = weeks(i);
  if isempty(week_datasets{1,wid})
  	fprintf('Ignoring invalid week %d',wid)
    continue;
  end

  % Build the features and labels:
  data = week_datasets{1,wid};
	assert(size(data,1)==cfg.num_week_minutes,'Unexpected number of rows in week dataset for week %d: n=%d',wid,size(data,1));

	week_prices = cfg.buildPriceMatrixFunc(data,cfg);
  week_features = cfg.buildFeatureMatrixFunc(data,cfg); 
  week_labels = cfg.buildLabelMatrixFunc(data,cfg); 

  % Append the data:
	assert(size(week_features,1)==nrows,'Unexpected number of rows in feature matrix for week %d: n=%d',wid,size(week_features,1));
	X = [X; week_features];

	assert(size(week_labels,1)==nrows,'Unexpected number of rows in label matrix for week %d: n=%d',wid,size(week_labels,1));	
	y = [y; week_labels];

	assert(size(week_prices,1)==nrows,'Unexpected number of rows in price matrix for week %d: n=%d',wid,size(week_prices,1));	
	prices = [prices; week_prices];
end

% Check if we should use rate of returns:
if cfg.use_rate_of_returns
	fprintf('Converting features to rate of returns...\n')

	% We have to divide each line by the previous line:
	X(2:end,2:end) = X(2:end,2:end) ./ X(1:end-1,2:end);

	% Remove the first line for all matrices:
	X(1,:) = [];
	y(1,:) = [];
	prices(1,:) = [];
end

% Check here if we should discard the number of minutes as features:
if cfg.discard_nmins_feature
	X(:,1) = [];
end

fprintf('Splitting datasets...\n')

% Now that we have the raw feature and label matrices, we have to separate them in train/cv/test sets using the set ratios 
% from the config:
ratios = cfg.dataset_ratios;
[X_train, X_cv, X_test] = splitDataset(X, ratios(1), ratios(2), ratios(3));
[y_train, y_cv, y_test] = splitDataset(y, ratios(1), ratios(2), ratios(3));
[prices_train, prices_cv, prices_test] = splitDataset(prices, ratios(1), ratios(2), ratios(3));

% Once the datasets are separated, we need to compute and apply feature normalization:
% The normalisation is computed fromt the train dataset only:
mu = mean(X_train);
sigma = std(X_train);

fprintf('Applying normalization...\n')

% Here we should keep a copy of the raw matrices:
trdata.X_train_raw = X_train;
trdata.X_cv_raw = X_cv;
trdata.X_test_raw = X_test;

% We also store the prices matrices:
trdata.prices_train = prices_train;
trdata.prices_cv = prices_cv;
trdata.prices_test = prices_test;

% Then we apply the normalization on all datasets:
X_train = applyNormalization(X_train,mu,sigma);
X_cv = applyNormalization(X_cv,mu,sigma);
X_test = applyNormalization(X_test,mu,sigma);

% fprintf('Done Applying normalization.\n')

% Once we have the initial training data matrix we can apply PCA if requested:
if cfg.use_PCA
	% First we need to build the covariance matrix:
	fprintf('Computing PCA...\n')
	Sigma = (X_train' * X_train)/size(X_train,1);
	if(cfg.verbose)
		fprintf('Performing PCA SVD decomposition...\n')
	end
	[U,S,V] = svd(Sigma);

	% fprintf('PCA S matrix size:\n')
	% size(S)

	% Compute the sum of all eigenvalues:
	eigvals = diag(S);
	total_eig = sum(eigvals);
	k=0;
	cur_val = 0;
	while cur_val< cfg.PCA_variance,
		k+=1;
		cur_val += 100.0 * eigvals(k)/total_eig;
	end
	fprintf('We need %d dimensions to keep %f%% of variance.\n',k,cfg.PCA_variance)

	% keep the reduction matrix in the training structure:
	reduce = U(:,1:k);
	trdata.reduce_mat = reduce;

	% Now we compute the projected data:
	X_train = applyPCAReduction(X_train,reduce);
	X_cv = applyPCAReduction(X_cv,reduce);
	X_test = applyPCAReduction(X_test,reduce);
end

% We may also want to shuffle the data here (for the training only):
if cfg.shuffle_training_data
	fprintf('Shuffling training data...\n')
	idx = randperm(size(X_train,1));
	X_train = X_train(idx,:);
	y_train = y_train(idx,:);
end

lpos = mean(y_train(:,cfg.target_symbol_pair)==1)*100.0;
spos = mean(y_train(:,cfg.target_symbol_pair)==2)*100.0;
npos = mean(y_train(:,cfg.target_symbol_pair)==0)*100.0;

% assert(abs(lpos+spos+npos - 100.0)< 1e-10,'Invalid label values, sum is: %.16f',lpos+spos+npos)

fprintf('Long position ratio: %f%%\n',lpos)
fprintf('Short position ratio: %f%%\n',spos)
fprintf('No position ratio: %f%%\n',npos)
fprintf('Building training structure...\n')

% Prepare the structure to return
trdata.X_train = X_train;
trdata.X_cv = X_cv;
trdata.X_test = X_test;
trdata.y_train = y_train;
trdata.y_cv = y_cv;
trdata.y_test = y_test;

trdata.num_features = size(X_train,2);

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

% RMS stop initial value:
trdata.rms_stop = cfg.default_rms_stop;

trdata.early_stopping = cfg.use_early_stopping;

trdata.learning_rate = cfg.default_learning_rate;

trdata.momentum = cfg.default_momentum;

trdata.validation_window_size = cfg.default_validation_window_size;

trdata.eval_frequency = cfg.default_eval_frequency;

trdata.mini_batch_size = 0;
end

% ==> Should return a struct containing an Xtrain matrix and Ytrain vector, both with same number of rows:

%!
%!test
%!	cfg = config();
%!	data = nnPrepareTraining(1:4,cfg);
%!	n1 = size(data.X_train,1);
%!	n2 = size(data.y_train,1);
%!	assert(n1==n2,'Mismatch between X_train and y_train')
%!	n1 = size(data.X_cv,1);
%!	n2 = size(data.y_cv,1);
%!	assert(n1==n2,'Mismatch between X_cv and y_cv')
%!	n1 = size(data.mu,2);
%!	n2 = size(data.sigma,2);
%!	assert(n1==n2,'Mismatch between mu and sigma')
%!	assert(n1==(1+4*cfg.num_symbol_pairs*cfg.num_input_bars),'Invalid col count in mu and sigma')
%!  assert(size(data.y_train,2)==cfg.num_symbol_pairs,'Invalid number of label colunms')
%! 	% Also check that the normalization is applied properly:
%!	Xn = data.X_train;
%!	assert(sum(abs(mean(Xn)))<1e-8,'Mean value is out of range: mean=%.16f',sum(abs(mean(Xn))))
%!	assert(cfg.use_PCA || sum(abs(std(Xn)-1.0))<1e-8,'Sigma is out of range: std=%.16f',sum(abs(std(Xn)-1.0)))
%!	assert(data.regularization_param==cfg.default_regularization_parameter,'Invalid regularization parameter value')
%!	assert(data.max_iterations==cfg.default_max_training_iterations,'Invalid max iterations value')
%!	assert(data.train_ratio==1.0,'Invalid train_ratio value')
%!	assert(data.deep_training==cfg.default_deep_training,'Invalid deep_training value')
