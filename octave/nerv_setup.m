% This script is used t demonstrate the global setup of the data
% needed to perform investigations.

% Initialization
clear; close all; clc
more off;
% tic()

% First we add the common location path:
pname = pwd();
% addpath(['/cygdrive/x/Station/CUDA_Toolkit-6.5/bin']); %add the binary folder.
arch=computer();
if strcmp(arch,'x86_64-w64-mingw32')==1
fprintf('Testing on x64 architecture.\n')
addpath([pname '/../bin/x64']); %add the binary folder.
else
fprintf('Testing on x86 architecture.\n')
addpath([pname '/../bin/x86']); %add the binary folder.
end

addpath([pname '/common']);
addpath([pname '/neural']);


% Load the global config object:
cfg = config();

% 1. We convert all the raw data files from .txt to .mat for fast loading.
% convertDataFiles(cfg.datapath,cfg.symbol_name);

% 2. Validating and synchronizing the week datasets:
% validateDatasetCollection(cfg)

% 3. Building the feature matrices for each week of interest.
% also building the label matrices at the same time.
% data = loadData(cfg.datapath,cfg.week_dataset_name,'week_data');
% buildFeatureMatrix(data,cfg);

% 4. Prepare a training set:
% cfg.use_PCA = true;
% cfg.shuffle_training_data = false;
% tr = nnPrepareTraining(1:1,cfg);	
% tr.early_stopping = true;
% tr.max_iterations = 100;
% fname = [cfg.datapath '/training_week_1_1_pca.mat'];
% save('-binary',fname,'tr');

% cfg.use_PCA = false;
% cfg.dataset_ratios = [0.60 0.20 0.20];
% tr = nnPrepareTraining(1:12,cfg);	
% tr.early_stopping = true;
% tr.max_iterations = 0;
% fname = [cfg.datapath '/training_weeks_1_12.mat'];
% save('-binary',fname,'tr');

cfg.use_PCA = false;
cfg.dataset_ratios = [0.4 0.1 0.5];
tr = nnPrepareTraining(1:24,cfg);	
tr.early_stopping = true;
tr.max_iterations = 0;
fname = [cfg.datapath '/training_weeks_1_24.mat'];
save('-binary',fname,'tr');

% tr = nnPrepareTraining(1:2,cfg);	
% tr.early_stopping = true;
% tr.max_iterations = 100;
% fname = [cfg.datapath '/training_weeks_1_2_pca.mat'];
% save('-binary',fname,'tr');

% tr = nnPrepareTraining(1:4,cfg);	
% tr.early_stopping = true;
% tr.max_iterations = 100;
% fname = [cfg.datapath '/training_weeks_1_4_pca.mat'];
% save('-binary',fname,'tr');

% cfg.use_PCA = false;
% tr = nnPrepareTraining(1:4,cfg);	
% tr.early_stopping = true;
% tr.max_iterations = 0;
% fname = [cfg.datapath '/training_weeks_1_4.mat'];
% save('-binary',fname,'tr');


% 5. Perform some training:
%	load training set:
% fname = [cfg.datapath '/training_weeks_1_4_pca.mat'];
% load(fname);
% nn = nnInitNetwork([tr.num_features 100 3],cfg);
% tic();
% nn = nnTrainNetworkCUDA(tr,nn,cfg);
% toc();


% 6. Write a trained network:
% fname = [cfg.datapath '/training_weeks_1_4.mat'];
% load(fname);
% tr.early_stopping = true;
% tr.max_iterations = 0;
% tr.dropouts = [0.8, 0.5, 0.5, 0.5];
% nn = nnInitNetwork([tr.num_features 512 128 32 3],cfg);
% nn = nnTrainNetworkNERV(tr,nn,cfg);
% nn.mu = tr.mu;
% nn.sigma = tr.sigma;
% fname = [cfg.datapath '/nn_512_128_32_3_drop_weeks_1_4.mat'];
% save('-binary',fname,'nn');
% nnEvaluateNetwork(tr,nn,cfg)

% % Write the figure:
% figure; hold on;
% h = gcf();	
% %nn.cost_iters;
% %nn.costs;
% plot(nn.cost_iters, nn.costs, 'LineWidth', 2, 'Color','b');
% legend('Jcv');
% title('Learning progress');
% xlabel('Number of epochs');
% ylabel('Cv Cost');
% hold off;


% 7. Once we have a network, we can build a strategy from it:


% week1_data = data{1,1};
% size(week1_data)

% Simple test to check the bad weeks:
% data = loadData(cfg.datapath,cfg.week_dataset_name,'week_data');
% bad_weeks = getMissingWeeks(data)

% Simple test to load feature matrix:
% data = loadData(cfg.datapath,sprintf(cfg.week_feature_pattern,1),'week_features');
% size(data)
% labels = loadData(cfg.datapath,sprintf(cfg.week_feature_pattern,1),'week_labels');
% size(labels)

% Simple test for the NN gradients computation:
% nnCheckGradients([3; 5; 3])

% toc()
more on;
