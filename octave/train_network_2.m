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

cfg = config();
% fname = [cfg.datapath '/training_weeks_1_12.mat'];
% load(fname);

% Prepare the config:
cfg.use_sparse_init = false;
cfg.use_PCA = false;
cfg.dataset_ratios = [0.60 0.20 0.20];

tr = nnPrepareTraining(1:12,cfg);	

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
tr.regularization_param = 0.01;

fname = [cfg.datapath '/training_weeks_1_12.mat'];
save('-binary',fname,'tr');

nn = nnInitNetwork([tr.num_features, 512, 3],cfg);
nn = nnTrainNetworkNERV(tr,nn,cfg);

% nn = nnSelectRandomNetwork(10,[512 3],tr,cfg,true)

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
accuracy = 1.0 - mean(double(origy ~= y))

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
fname = [cfg.datapath '/nn_512_3_weeks_1_12.mat'];
save('-binary',fname,'nn');

more on;
