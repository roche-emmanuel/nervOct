% Initial test for neural network usage.

% Initialization
clear; close all; clc
more off;

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

% fprintf('Testing nnSparseInitializeWeights...\n')
% test nnSparseInitializeWeights

% fprintf('Testing nnCostFunction...\n')
% test nnCostFunction

% fprintf('Testing nn_cost_function...\n')
% test tests_nn_cost_function;

% fprintf('Testing nn_cost_function_cuda...\n')
% test tests_nn_cost_function_cuda;

% fprintf('Testing splitDataset...\n')
% test splitDataset

% fprintf('Testing applyNormalization...\n')
% test applyNormalization

% fprintf('Testing applyPCAReduction...\n')
% test applyPCAReduction

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

% fprintf('Testing nn_cg_train_cpu...\n')
% test tests_nn_cg_train_cpu;

% fprintf('Testing nn_cg_train...\n')
% test tests_nn_cg_train;

% fprintf('Testing nnTrainNetworkCUDA...\n')
% test nnTrainNetworkCUDA

% fprintf('Testing nnTrainNetworkNERV...\n')
% test nnTrainNetworkNERV

% fprintf('Testing nnPredict...\n')
% test nnPredict

% fprintf('Testing nnEvaluateNetwork...\n')
% test nnEvaluateNetwork

% fprintf('Testing DLL loading...\n')
% test_load

% fprintf('Showing CUDA info...\n')
% show_cuda_info

% fprintf('Testing train_bp...\n')
% test tests_train_bp;

fprintf('Testing Gradient descent...\n')
test tests_nn_gradient_descent

% fprintf('Testing nnSelectRandomNetwork...\n')
% test nnSelectRandomNetwork

% fprintf('Testing nnRescaleParameters...\n')
% test nnRescaleParameters

% fprintf('Testing nnComputeLearningCurves...\n')
% test nnComputeLearningCurves

% fprintf('Testing Trade Strategy...\n')
% test tests_trade_strategy

% fprintf('Testing Training...\n')
% test tests_training_1
% test tests_training_2
% test tests_training_2b
% test tests_training_3
% test tests_training_4

more on;
