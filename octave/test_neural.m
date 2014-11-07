% Initial test for neural network usage.

% Initialization
clear; close all; clc
more off;

% First we add the common location path:
pname = pwd();
addpath([pname '/common']);

% Load the default configuration for the test:
options = config();

fprintf('Loading file %s...\n',options.datafile)
options.train_data = load(["../../data/" options.datafile]);

fprintf('Loading file %s...\n',options.testfile)
options.test_data = load(["../../data/" options.testfile]);

% Setup of the callbacks:
options.genFeatureFunc=@buildFeatureMatrix;
options.genLabelFunc=@buildLabelVector;
% options.genTestFeatureFunc= To be defined.
% options.tradeEvalFunc= To be defined.

% buildClassifier(options);

evaluateLearning(10,options);

more on;
