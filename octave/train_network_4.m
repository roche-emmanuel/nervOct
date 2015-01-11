% In this training we try to first build a layer of hidden features,
% then we had a solfmax layer on top of it.

% Initialization
clear; close all; clc
more off;
% tic()

% First we add the common location path:
pname = pwd();
% addpath(['/cygdrive/x/Station/CUDA_Toolkit-6.5/bin']); %add the binary folder.
arch=computer();
if strcmp(arch,'x86_64-w64-mingw32')==1
fprintf('Running on x64 architecture.\n')
addpath([pname '/../bin/x64']); %add the binary folder.
else
fprintf('Running on x86 architecture.\n')
addpath([pname '/../bin/x86']); %add the binary folder.
end

addpath([pname '/common']);
addpath([pname '/neural']);
addpath([pname '/ufldl/minFunc']);
addpath([pname '/ufldl/ex5']);
addpath([pname '/ufldl/ex4']);
addpath([pname '/ufldl/ex2']);

function np = compute_np(lsizes)
	np = 0;
	nt = numel(lsizes)-1;
	for i=1:nt
		np += (lsizes(i)+1)*lsizes(i+1);
	end
end

function str = rangeToString(range)
	mini = min(range);
	maxi = max(range);
	str = [int2str(mini) '_' int2str(maxi)];
end

function str = lsizesToString(lsizes)
	str = [int2str(lsizes(1))];
	num = numel(lsizes);
	for i=2:num
		str = [str '_' int2str(lsizes(i))];
	end
end

function data = normalizeData(data,m,dev)
	% Remove DC (mean of images). 
	data = data - m;

	% Truncate to +/-3 standard deviations and scale to -1 to 1
	pstd = 3 * dev;
	data = max(min(data, pstd), -pstd) / pstd;

	% Rescale from [-1,1] to [0.1,0.9]
	data = (data + 1) * 0.4 + 0.1;
end

% Testing week range:
trange = 1:12;
plsizes = [256 3];

cfg = config();
% fname = [cfg.datapath '/training_weeks_1_12.mat'];
% load(fname);

% Prepare the config:
cfg.num_input_bars=120;
cfg.num_pred_bars=5;
cfg.use_sparse_init = false;
cfg.use_PCA = false;
cfg.dataset_ratios = [0.60 0.20 0.20];
cfg.use_rate_of_returns = true;
cfg.discard_nmins_feature = true;

tr = nnPrepareTraining(trange,cfg);	

% fprintf('Range string is %s.\n',rangeToString(1:12))

% tr.X_train_raw(1:20,1:20)
% error('Testing');

tr.train_ratio = 1.0;
tr.early_stopping = true;
tr.learning_rate = 0.0001;
tr.momentum = 0.99;
tr.verbose = true;
tr.max_iterations = 40000;
tr.eval_frequency = 32;
tr.mini_batch_size = 128;
tr.validation_window_size = 100;
tr.min_cost_decrease = 0.0;
tr.learning_decay = 0.9999;
tr.regularization_param = 0.001;
tr.ping_frequency = 500;
tr.with_softmax = true;
% tr.dropouts = [0.8 0.5];

fname = [cfg.datapath '/training_weeks_' rangeToString(trange) '.mat'];
save('-binary',fname,'tr');


%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

inputSize = tr.num_features;
hiddenSize = plsizes(1);
beta = 3.0;
sparsityParam = 0.1;

trainFeatures = tr.X_train';

% Renormalize the trainFeatures between 0 and 1 to accept auto encoding:
mm = mean(trainFeatures(:));
dev = std(trainFeatures(:));

fprintf('Features mean: %f, deviation: %f\n', mm, dev);

trainFeatures = normalizeData(trainFeatures,mm,dev);

% labels should start from 1 here:
trainLabels = tr.y_train(:,cfg.target_symbol_pair)+1;

% Compute the predictions:
testFeatures = tr.X_cv';
testFeatures = normalizeData(testFeatures,mm,dev);

testLabels = tr.y_cv(:,cfg.target_symbol_pair)+1;


%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

% Train the network:
desc.lsizes = [inputSize hiddenSize inputSize];
desc.X_train = trainFeatures;
desc.y_train = trainFeatures;
desc.params = theta;
desc.lambda = tr.regularization_param;
desc.useSoftmax = false;
desc.spaeBeta = beta;
desc.spaeSparsity = sparsityParam;

% Register the device resources:
desc.id = nn_create_traits(desc);

%  Use minFunc to minimize the function
% options.Method = 'sd'; % Here, we use L-BFGS to optimize our cost
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
options.useMex = 0;

% options = struct('GradObj','on','Display','iter','MaxIter',400);
% options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','bfgs','InitialHessType','identity','GoalsExactAchieve',0);
% options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','bfgs','InitialHessType','identity','GoalsExactAchieve',1,'GradConstr',false);

tic()

opttheta = theta;

[opttheta, cost] = minFunc( @(p) spae_costfunc_dev(p,desc), theta, options);
% [opttheta, cost] = minFunc( @(p) spae_costfunc(p,desc), theta, options);

% [opttheta, cost] = fminunc( @(p) sparseAutoencoderCost(p, ...
% [opttheta, cost] = fminlbfgs( @(p) sparseAutoencoderCost(p, ...
% [opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
%                                    inputSize, hiddenSize, ...
%                                    lambda, sparsityParam, ...
%                                    beta, patches), ...
%                             theta, options);

toc()

% unregister the device resources:
nn_destroy_traits(desc.id);



%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, trainFeatures);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, testFeatures);


%%======================================================================
%% STEP 4: Train the softmax classifier

tic()
lambda = 0.01;
X = [ones(1,size(trainFeatures,2)); trainFeatures];

% Now we train a simple softmax layer on top of our features:
options.maxIter = 400;

softmaxModel = softmaxTrain(hiddenSize+1, plsizes(end), lambda, X, trainLabels, options);
toc()


%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

Xtest = [ones(1,size(testFeatures,2)); testFeatures];

pred = softmaxPredict(softmaxModel, Xtest);

acc = mean(testLabels(:) == pred(:));
fprintf('Num training samples: %d\n', size(trainFeatures,2))
fprintf('Num testing samples: %d\n', size(testFeatures,2))

fprintf('Accuracy: %0.3f%%\n', acc * 100);

more on;
