% Build a classifier taking as input the raw train data,
% and a method used to generate the feature matrix from the raw data.
% genLabelFunc is used to generate the label vector from the raw data.
function [theta,mu,sigma,J_train,J_test,train_m, X_test, y_test] = buildClassifier(options)
	% Build the feature matrix:
	% fprintf('Building feature matrix...\n')
	% tic();
	[X, doffset] = options.genFeatureFunc(options.train_data,options);
	% toc();

	% Generate the labels:
	% fprintf('Building label vector...\n')
	% tic();
	y = options.genLabelFunc(options.train_data,doffset,options);
	% toc();

	% Once we have the feature matrix and the label vector, we need to shuffle the indexes and split into train/cv/test sets:
	% fprintf('Splitting dataset...\n')
	% tic();
	[X_train, y_train, X_test, y_test, X_cv, y_cv] = splitDataset(X,y,options.train_ratio,0.75,0.25); % Only prepare train dataset.
	% toc();
	train_m = size(X_train,1);
	test_m = size(X_test,1);

	% fprintf('train labels:\n')
	% y_train
	% fprintf('test labels:\n')
	% y_test

	% Normalize the training set:
	% fprintf('Normalizing feature matrix...\n')
	% tic();
	[mu, sigma] = computeFeatureNormalization(X_train);
	X_train = normalizeFeatureMatrix(X_train,mu,sigma);
	X_test = normalizeFeatureMatrix(X_test,mu,sigma);
	% toc();

	% Add the intercept term:
	X_train = [ones(size(X_train,1), 1) X_train];
	X_test = [ones(size(X_test,1), 1) X_test];

	% Train the classifier
	fprintf('Training multi-classifier...\n')
	tic();
	theta = trainClassifier(X_train,y_train,3); % We have 3 classes.
	toc();

	% Evaluation the classifier:
	% fprintf('Evaluating classifier...\n')
	% tic()
	% First on the train data:
	[p, prob] = predictLabel(theta, X_train);
	acc = mean(double(p == y_train)) * 100;
	fprintf('Train Accuracy: %.2f\n', acc);
	J_train = 100 - acc;
	% fprintf('Train buy predictions: %.2f\n', mean(double(p==2)) * 100);
	% fprintf('Train sell predictions: %.2f\n', mean(double(p==3)) * 100);

	% Then on the test data:
	[p, prob] = predictLabel(theta, X_test);
	acc = mean(double(p == y_test)) * 100;
	fprintf('Test Accuracy: %.2f\n', acc);
	J_test = 100 - acc;
	% fprintf('Test buy predictions: %.2f\n', mean(double(p==2)) * 100);
	% fprintf('Test sell predictions: %.2f\n', mean(double(p==3)) * 100);
	% toc();
end
