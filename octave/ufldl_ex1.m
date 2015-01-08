% This script is used t demonstrate the global setup of the data
% needed to perform investigations.

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
addpath([pname '/ufldl/minFunc']);
addpath([pname '/ufldl/ex1']);

%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

visibleSize = 8*8;   % number of input units 
hiddenSize = 25;     % number of hidden units 
sparsityParam = 0.01;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term       


%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
%  After implementing sampleIMAGES, the display_network command should
%  display a random sample of 200 patches from the dataset

patches = sampleIMAGES;
display_network(patches(:,randi(size(patches,2),200,1)));

%  Obtain random parameters theta
theta = initializeParameters(hiddenSize, visibleSize);

% Train the network:
desc.lsizes = [visibleSize hiddenSize visibleSize];
desc.X_train = patches;
desc.y_train = patches;
desc.params = theta;
desc.lambda = lambda;
desc.useSoftmax = false;
desc.spaeBeta = beta;
desc.spaeSparsity = sparsityParam;
% desc.costMode = 3; % 3 => COST_RMS. % We should not need to specify this if beta > 0.

% Register the device resources:
desc.id = nn_create_traits(desc);

% [nn_cost, nn_grad] = nn_costfunc(desc);


%  Use minFunc to minimize the function
% options.Method = 'sd'; % Here, we use L-BFGS to optimize our cost
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
options.useMex = 0;

% options = struct('GradObj','on','Display','iter','MaxIter',400);
% options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','bfgs','InitialHessType','identity','GoalsExactAchieve',0);
% options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','bfgs','InitialHessType','identity','GoalsExactAchieve',1,'GradConstr',false);

tic()

[opttheta, cost] = minFunc( @(p) spae_costfunc_dev(p,desc), theta, options);
% [opttheta, cost] = minFunc( @(p) spae_costfunc(p,desc), theta, options);

% [opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
% [opttheta, cost] = fminunc( @(p) sparseAutoencoderCost(p, ...
% [opttheta, cost] = fminlbfgs( @(p) sparseAutoencoderCost(p, ...
                              %      visibleSize, hiddenSize, ...
                              %      lambda, sparsityParam, ...
                              %      beta, patches), ...
                              % theta, options);

toc()

% unregister the device resources:
nn_destroy_traits(desc.id);

%%======================================================================
%% STEP 5: Visualization 

W1 = reshape(opttheta(1:hiddenSize*(visibleSize+1)), hiddenSize, visibleSize+1);
% Drop the first column:
W1(:,1) = [];
display_network(W1', 12); 

print -djpeg 'ufldl/ex1/weights.jpg'   % save the visualization to a file 


%%======================================================================
%% STEP 5: Visualization 

% W1 = reshape(weights(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% display_network(W1', 12); 

% print -djpeg weights.jpg   % save the visualization to a file 

% Print the costs:
% figure; hold on;
% h = gcf();	
% plot(iters, costs, 'LineWidth', 2, 'Color','b');
% legend('Jcv');
% title('Learning progress');
% xlabel('Number of epochs');
% ylabel('Cv Cost');
% hold off;
% fprintf('Final cost is: %f.\n',Jcv)

more on;
