% Initial test for neural network usage.

% Initialization
clear; close all; clc
more off;

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

% fprintf('Testing Gradient descent...\n')
% test tests_nn_gradient_descent

% fprintf('Testing nnSelectRandomNetwork...\n')
% test nnSelectRandomNetwork

% fprintf('Testing nnRescaleParameters...\n')
% test nnRescaleParameters

% fprintf('Testing nnComputeLearningCurves...\n')
% test nnComputeLearningCurves

% fprintf('Testing Trade Strategy...\n')
% test tests_trade_strategy
% test tests_trade_strategy_2

% fprintf('Testing Training...\n')
% test tests_training_1
% test tests_training_2
% test tests_training_2b
% test tests_training_3
% test tests_training_4
% test tests_training_5
% test tests_training_6
% test tests_training_7
% test tests_training_8
% test tests_training_9
% test tests_training_10

% fprintf('Testing traits creation...\n')
% test tests_create_traits

% fprintf('Testing train accuracy...\n')
% test_train_accuracy
% test_train_accuracy_2
% test_train_accuracy_3
% test_train_accuracy_4

% test_multi_accuracy

fprintf('Testing create_simple_strategy...\n')
test create_simple_strategy

fprintf('Testing create_base_model...\n')
test create_base_model

fprintf('Testing create_rand_model...\n')
test create_rand_model

fprintf('Testing create_lreg_model...\n')
test create_lreg_model


more on;
