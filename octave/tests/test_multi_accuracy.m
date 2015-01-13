% In this test we check if we can predict a simple label properly:

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
plsizes = [256 3];

cfg = config();
% fname = [cfg.datapath '/training_weeks_1_12.mat'];
% load(fname);

% Prepare the config:
cfg.num_input_bars=120;
cfg.num_pred_bars=5;
cfg.use_sparse_init = true;
cfg.use_PCA = true;
cfg.PCA_variance = 99.0;
cfg.dataset_ratios = [0.60 0.20 0.20];
cfg.label_buy_sell_delta = 0.00015;
cfg.min_gain=cfg.spread*4.0;
cfg.use_rate_of_returns = true;
cfg.shuffle_training_data = true;
cfg.discard_nmins_feature = true;
cfg.buildFeatureMatrixFunc = @buildWeekFeatureMatrix_C;
% cfg.buildLabelMatrixFunc = @buildWeekLabelMatrix_debug;
% cfg.buildLabelMatrixFunc = @buildWeekLabelMatrix_debug2;
cfg.buildLabelMatrixFunc = @buildWeekLabelMatrix_AtTime;

tr = nnPrepareTraining(trange,cfg);	

tr.train_ratio = 1.0;
tr.early_stopping = true;
tr.learning_rate = 0.0001;
tr.momentum = 0.99;
tr.verbose = true;
tr.max_iterations = 30000;
tr.eval_frequency = 32;
tr.mini_batch_size = 128;
tr.validation_window_size = 100;
tr.min_cost_decrease = 0.0;
% tr.learning_decay = 0.9999;
tr.learning_decay = 1.0;
tr.regularization_param = 0.01; %0.001;
tr.ping_frequency = 500;
tr.with_softmax = true;
% tr.dropouts = [0.8 0.5];

% iterate on the number of networks we want to build:
numnet = 4;
lsizes = [tr.num_features plsizes]

trainLabels = tr.y_train(:,cfg.target_symbol_pair);
cvLabels = tr.y_cv(:,cfg.target_symbol_pair);

fprintf('Train dataset: %d None labels. %d Buy labels. %d Sell labels.\n',sum(trainLabels==0),sum(trainLabels==1),sum(trainLabels==2));
fprintf('CV dataset: %d None labels. %d Buy labels. %d Sell labels.\n',sum(cvLabels==0),sum(cvLabels==1),sum(cvLabels==2));

pred_train = [];
pred_cv = [];

taccs = zeros(numnet,1);
cvaccs = zeros(numnet,1);

% ratio of transactions
trtrans = zeros(numnet,1);
cvrtrans = zeros(numnet,1);

for i=1:numnet
	% train a new network:
	nn = nnInitNetwork(lsizes,cfg);
	nn = nnTrainNetworkNERV(tr,nn,cfg);

	% Compute the predictions:
	yn_train = nnPredict(nn,tr.X_train);
	yn_cv = nnPredict(nn,tr.X_cv);

	% Stack the predictions on the prediction matrices:
	pred_train = [pred_train yn_train];
	pred_cv = [pred_cv yn_cv];

	% Now compute an overall prediction:
	ns = size(yn_train,1);
	y_train = zeros(ns,1);
	y_train += (!any(pred_train-1,2))*1;
	y_train += (!any(pred_train-2,2))*2;

	ns = size(yn_cv,1);
	y_cv = zeros(ns,1);
	y_cv += (!any(pred_cv-1,2))*1;
	y_cv += (!any(pred_cv-2,2))*2;

	% Compute some ratios:
	acc = (sum(y_train==1 & trainLabels==1)+sum(y_train==2 & trainLabels==2))/(sum(y_train==1)+sum(y_train==2));
	rt = (sum(y_train==1)+sum(y_train==2))/numel(y_train);
	fprintf('Train trade accuracy: %0.2f%%\n', acc * 100);
	fprintf('Train transaction ratio: %0.2f%% (%d)\n', rt * 100,sum(y_train==1)+sum(y_train==2));
	taccs(i) = acc;
	trtrans(i) = rt;
	acc = (sum(y_cv==1 & cvLabels==1)+sum(y_cv==2 & cvLabels==2))/(sum(y_cv==1)+sum(y_cv==2));
	rt = (sum(y_cv==1)+sum(y_cv==2))/numel(y_cv);
	fprintf('CV trade accuracy: %0.2f%%\n', acc * 100);
	fprintf('CV transaction ratio: %0.2f%% (%d)\n', rt * 100,sum(y_cv==1)+sum(y_cv==2));
	cvaccs(i) = acc;
	cvrtrans(i) = rt;
end	

% Now we can draw the evolution of the costs:
figure; hold on;
h = gcf();	
plot(1:numnet, taccs, 'LineWidth', 2, 'Color','b');
plot(1:numnet, cvaccs, 'LineWidth', 2, 'Color','r');
plot(1:numnet, trtrans, 'LineWidth', 2, 'Color','g');
plot(1:numnet, cvrtrans, 'LineWidth', 2, 'Color','k');
legend('Train accuracy', 'CV accuracy','Train transactions','CV transactions');
title('Trade accuracy');
xlabel('Number of networks');
ylabel('Accuracy');
hold off;

more on;
