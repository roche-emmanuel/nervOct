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
tr.max_iterations = 50000;
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

lsizes = [tr.num_features plsizes]
nn = nnInitNetwork(lsizes,cfg);
nn = nnTrainNetworkNERV(tr,nn,cfg);

% nn = nnSelectRandomNetwork(10,plsizes,tr,cfg,true)

X_train = tr.X_train;
y_train = nnBuildLabelMatrix(tr.y_train(:,cfg.target_symbol_pair))';

[y yy] = nnPredict(nn,X_train);
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
fprintf('Accuracy: %0.3f%%\n', acc * 100);

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

assert(sum(sum(prec_recall))==numel(y),'Invalid prec_recal count: %d!=%d',sum(sum(prec_recall)), numel(y))

	% Now we can draw the evolution of the costs:
figure; hold on;
h = gcf();	
plot(nn.cost_iters, nn.costs, 'LineWidth', 2, 'Color','b');
legend('Jcv');
title('Trade Week Learning progress');
xlabel('Number of epochs');
ylabel('Cv Cost');
hold off;

% Now we save the generated network:
nn.mu = tr.mu;
nn.sigma = tr.sigma;
fname = [cfg.datapath '/nn_' lsizesToString(plsizes) '_weeks_' rangeToString(trange) '.mat'];
save('-binary',fname,'nn');

more on;
