% This script is used t demonstrate the global setup of the data
% needed to perform investigations.

% Initialization
clear; close all; clc
more off;
tic()

% First we add the common location path:
pname = pwd();
addpath([pname '/common']);

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

toc()
more on;
