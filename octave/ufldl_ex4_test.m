% This script is used t demonstrate the global setup of the data
% needed to perform investigations.

% Initialization
clear; close all; clc
more off;

% First we add the common location path:
pname = pwd();

arch=computer();
if strcmp(arch,'x86_64-w64-mingw32')==1
fprintf('Testing on x64 architecture.\n')
addpath([pname '/../bin/x64']); %add the binary folder.
else
fprintf('Testing on x86 architecture.\n')
addpath([pname '/../bin/x86']); %add the binary folder.
end

% addpath([pname '/common']);
% addpath([pname '/neural']);
addpath([pname '/ufldl/minFunc']);
addpath([pname '/ufldl/ex4']);
addpath([pname '/ufldl/ex2']);

% Compute the softmax cost
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)

lambda = 1e-4; % Weight decay parameter

inputSize = 8;
inputData = [ones(1,100); randn(7, 100)];
labels = randi(10, 100, 1);

% Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize, 1);

%%======================================================================
%% STEP 2: Implement softmaxCost
%
%  Implement softmaxCost in softmaxCost.m. 

[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);

fprintf('Octave cost: %f,\n',cost);

% Now compute the same thing on GPU:
desc.lsizes = [(inputSize-1) numClasses];
desc.X_train = inputData(2:end,:); % We do not provide the intercept term row.
groundTruth = full(sparse(labels, 1:100, 1));
desc.y_train = groundTruth;
desc.params = theta;
desc.lambda = lambda;
desc.useSoftmax = true;
% desc.costMode = 2; % 2 => COST_SOFTMAX.

desc.id = nn_create_traits(desc);

[nn_cost, nn_grad] = nn_costfunc_device(desc);

nn_destroy_traits(desc.id);

% Compare the values:
assert(abs(cost-nn_cost) < 1e-12, 'Mismatch in cost values: %.16f != %.16f', cost, nn_cost);

disp([nn_grad grad]); 

diff = norm(nn_grad-grad)/norm(nn_grad+grad);
disp(diff); % Should be small. In our implementation, these values are
assert(diff< 1e-10,'Mismatch in computed gradients')


more on;
