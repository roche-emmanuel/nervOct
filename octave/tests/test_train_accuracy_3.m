% In this test we check if we can predict a simple label properly:
% Here we try to combine the predictions of 2 simple models.

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
cfg.use_PCA = false;
cfg.dataset_ratios = [0.60 0.20 0.20];
cfg.use_rate_of_returns = true;
cfg.shuffle_training_data = true;
cfg.discard_nmins_feature = true;
cfg.buildFeatureMatrixFunc = @buildWeekFeatureMatrix_C;
% cfg.buildLabelMatrixFunc = @buildWeekLabelMatrix_debug;
cfg.buildLabelMatrixFunc = @buildWeekLabelMatrix_debug2;

tr = nnPrepareTraining(trange,cfg);	

% fprintf('Range string is %s.\n',rangeToString(1:12))

% tr.X_train_raw(1:20,1:20)
% error('Testing');

tr.train_ratio = 1.0;
tr.early_stopping = true;
tr.learning_rate = 0.0001;
tr.momentum = 0.99;
tr.verbose = true;
tr.max_iterations = 50000;
tr.eval_frequency = 32;
tr.mini_batch_size = 128;
tr.validation_window_size = 100;
tr.min_cost_decrease = 0.0;
% tr.learning_decay = 0.9999;
tr.learning_decay = 1.0;
tr.regularization_param = 0.001; %0.001;
tr.ping_frequency = 500;
tr.with_softmax = true;
% tr.dropouts = [0.8 0.5];

% Do not save the training set here:
% fname = [cfg.datapath '/training_weeks_' rangeToString(trange) '.mat'];
% save('-binary',fname,'tr');

lsizes = [tr.num_features plsizes]

X_train = tr.X_train;
labels = tr.y_train(:,cfg.target_symbol_pair);

% Now check that the labels really are what they should be:
% tr.X_train_raw(1:20,1:12)

% diff = tr.X_train_raw(:,cfg.target_symbol_pair); % - tr.X_train_raw(:,2*cfg.target_symbol_pair);

% % diff(1:20)

% plabels = zeros(size(X_train,1),1);

% plabels += (diff >= 1.00005)*1; % 1 is the id for 'buy'
% plabels += (diff <= 0.99995)*2; % 2 is the id for 'sell'

% plabels(end-20:end)
% labels(end-20:end)

% miss = plabels!=labels;
% fprintf('Number of labels: %d\n',size(plabels,1));

% list = 1:size(plabels,1);
% list(miss)
% diff(miss)
% plabels(miss)
% labels(miss)

% mml = max(max(plabels));
% assert(mml==2,'Invalid max label value: %d',mml)

% There can still be some errors here due to the way the label is computed for the last element of the week.
% len = sum(abs(plabels-labels));
% assert(len==0.0,'Mismatch in computed labels: %f != 0',len);
% fprintf('Debug labels are OK!\n');

fprintf('Found %d None labels.\n',sum(labels==0))
fprintf('Found %d Buy labels.\n',sum(labels==1))
fprintf('Found %d Sell labels.\n',sum(labels==2))

% Perform the actual training:
nn1 = nnInitNetwork(lsizes,cfg);
nn1 = nnTrainNetworkNERV(tr,nn1,cfg);

% Now train the second network:
nn2 = nnInitNetwork(lsizes,cfg);
nn2 = nnTrainNetworkNERV(tr,nn2,cfg);

nn3 = nnInitNetwork(lsizes,cfg);
nn3 = nnTrainNetworkNERV(tr,nn3,cfg);


% nn = nnSelectRandomNetwork(10,plsizes,tr,cfg,true)

y_train = nnBuildLabelMatrix(labels)';

[y1 yy] = nnPredict(nn1,X_train);
[y2 yy] = nnPredict(nn2,X_train);
[y3 yy] = nnPredict(nn3,X_train);

% Now build the mixed predictions:
y = zeros(size(y1,1),1);
y += (y1==1 & y2==1 & y3==1)*1;
y += (y1==2 & y2==2 & y3==2)*2;

[dummy, origy] = max(y_train', [], 2);
origy = origy-1;

% origy(1:20,:)
% y(1:20,:)
% yy(1:20,:)
pred_none_count = sum(y==0)
pred_buy_count = sum(y==1)
pred_sell_count = sum(y==2)

% compute buy precision:
acc = 1.0 - mean(double(origy ~= y));
fprintf('Accuracy: %0.2f%%\n', acc * 100);

tacc = (sum(y==1 & origy==1)+sum(y==2 & origy==2))/(sum(y==1)+sum(y==2));
fprintf('Trade accuracy: %0.2f%%\n', tacc * 100);

% prediction/ original data precision and recall:
none_none = sum(y==0 & origy==0);
none_buy = sum(y==0 & origy==1);
none_sell = sum(y==0 & origy==2);
buy_none = sum(y==1 & origy==0);
buy_buy = sum(y==1 & origy==1);
buy_sell = sum(y==1 & origy==2);
sell_none = sum(y==2 & origy==0);
sell_buy = sum(y==2 & origy==1);
sell_sell = sum(y==2 & origy==2);

prec_recall = [none_none none_buy none_sell;
buy_none buy_buy buy_sell;
sell_none sell_buy sell_sell]

% Compute the buy and sell efficiency and the mean efficiency:
buy_eff = buy_buy/(buy_none+buy_buy+buy_sell);
sell_eff = sell_sell/(sell_none+sell_buy+sell_sell);
mean_eff = (buy_buy+sell_sell)/(buy_none+buy_buy+buy_sell+sell_none+sell_buy+sell_sell);

fprintf('Buy efficiency: %.2f%%\n',buy_eff*100.0);
fprintf('Sell efficiency: %.2f%%\n',sell_eff*100.0);
fprintf('Mean efficiency: %.2f%%\n',mean_eff*100.0);

assert(sum(sum(prec_recall))==numel(y),'Invalid prec_recal count: %d!=%d',sum(sum(prec_recall)), numel(y))

% Now we can draw the evolution of the costs:
figure; hold on;
h = gcf();	
plot(nn2.cost_iters, nn2.costs, 'LineWidth', 2, 'Color','b');
legend('Jcv');
title('Trade Week Learning progress');
xlabel('Number of epochs');
ylabel('Cv Cost');
hold off;

% Now we save the generated network:
nn1.mu = tr.mu;
nn1.sigma = tr.sigma;
% fname = [cfg.datapath '/nn_' lsizesToString(plsizes) '_weeks_' rangeToString(trange) '.mat'];
fname = [cfg.datapath '/nn1.mat'];
save('-binary',fname,'nn1');

nn2.mu = tr.mu;
nn2.sigma = tr.sigma;
% fname = [cfg.datapath '/nn_' lsizesToString(plsizes) '_weeks_' rangeToString(trange) '.mat'];
fname = [cfg.datapath '/nn2.mat'];
save('-binary',fname,'nn2');

more on;

% results:
% Buy efficiency: 56.81%
% Sell efficiency: 61.10%
% Mean efficiency: 58.52%
