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
% tr = nnPrepareTraining(1:1,cfg);	
% tr.early_stopping = true;
% tr.max_iterations = 100;
% fname = [cfg.datapath '/training_week1_pca.mat'];
% save('-binary',fname,'tr');

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


% 5. Perform some training:
%	load training set:
fname = [cfg.datapath '/training_weeks_1_4_pca.mat'];
load(fname);
nn = nnInitNetwork([tr.num_features 100 3]);
tic();
nn = nnTrainNetworkCUDA(tr,nn,cfg);
toc();


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
