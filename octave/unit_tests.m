% Initial test for neural network usage.

% Initialization
clear; close all; clc
more off;

% First we add the common location path:
pname = pwd();
% addpath(['/cygdrive/x/Station/CUDA_Toolkit-6.5/bin']); %add the binary folder.
addpath([pname '/../bin/x86']); %add the binary folder.
addpath([pname '/common']);
addpath([pname '/neural']);

% fprintf('Testing sigmoid...\n')
% test sigmoid

% fprintf('Testing loadData...\n')
% test loadData

% fprintf('Testing validateDataset...\n')
% test validateDataset

% fprintf('Testing buildWeekFeatureMatrix...\n')
% test buildWeekFeatureMatrix

% fprintf('Testing buildWeekLabelMatrix...\n')
% test buildWeekLabelMatrix

% fprintf('Testing nnRandInitializeWeights...\n')
% test nnRandInitializeWeights

% fprintf('Testing nnDebugInitializeWeights...\n')
% test nnDebugInitializeWeights

% fprintf('Testing nnCostFunction...\n')
% test nnCostFunction

% fprintf('Testing splitDataset...\n')
% test splitDataset

% fprintf('Testing applyNormalization...\n')
% test applyNormalization

% fprintf('Testing nnPrepareTraining...\n')
% test nnPrepareTraining

% fprintf('Testing nnPrepareDebugTraining...\n')
% test nnPrepareDebugTraining

% fprintf('Testing nnInitNetwork...\n')
% test nnInitNetwork

% fprintf('Testing nnBuildLabelMatrix...\n')
% test nnBuildLabelMatrix

% fprintf('Testing nnTrainNetwork...\n')
% test nnTrainNetwork

% fprintf('Testing nnPredict...\n')
% test nnPredict

% fprintf('Testing nnEvaluateNetwork...\n')
% test nnEvaluateNetwork

% fprintf('Testing nnComputeLearningCurves...\n')
% test nnComputeLearningCurves

% fprintf('Testing DLL loading...\n')
% test_load

fprintf('Testing train_bp...\n')
test tests_train_bp;

more on;
