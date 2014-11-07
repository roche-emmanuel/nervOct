% This is a collection of functions to perform logistic regression in forex trading.

% Initialization
clear; close all; clc
more off;

% First we add the common location path:
pname = pwd();
addpath([pname '/common']);
addpath([pname '/logistic']);

% Load the default configuration for the test:
options = config();

% Setup of the callbacks:
options.genFeatureFunc=@buildFeatureMatrix;
options.genLabelFunc=@buildLabelVector;
% options.genTestFeatureFunc= To be defined.
% options.tradeEvalFunc= To be defined.

% buildClassifier(options);

evaluateLearning(2,options);

more on;
