% In this training we try to use only a softmax layer to decide the output.

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

% Testing week range:
trange = 1:12;
plsizes = [512 3];

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
tr.regularization_param = 0.1;
tr.ping_frequency = 500;
tr.with_softmax = true;
% tr.dropouts = [0.8 0.5];

fname = [cfg.datapath '/training_weeks_' rangeToString(trange) '.mat'];
save('-binary',fname,'tr');

% Now we train a simple softmax layer on top of our features:

options.maxIter = 400;

% Train the network:
tic()

trainFeatures = tr.X_train';
% Add a row of ones:
trainFeatures = [ ones(1,size(trainFeatures,2)); trainFeatures];

% labels should start from 1 here:
trainLabels = tr.y_train(:,cfg.target_symbol_pair)+1;


softmaxModel = softmaxTrain(tr.num_features+1, plsizes(end), tr.regularization_param, trainFeatures, trainLabels, options);
toc()

% Compute the predictions:
testFeatures = tr.X_cv';
% Add a row of ones:
testFeatures = [ ones(1,size(testFeatures,2)); testFeatures];

testLabels = tr.y_cv(:,cfg.target_symbol_pair)+1;

pred = softmaxPredict(softmaxModel, testFeatures);

acc = mean(testLabels(:) == pred(:));
fprintf('Num training samples: %d\n', size(trainFeatures,2))
fprintf('Num testing samples: %d\n', size(testFeatures,2))

fprintf('Accuracy: %0.3f%%\n', acc * 100);

more on;
