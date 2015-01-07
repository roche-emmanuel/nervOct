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
lambda = 0.0; %0.0001;     % weight decay parameter       
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

[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, patches);

fprintf('Octave cost is: %f.\n',cost)


% Train the network:
desc.lsizes = [visibleSize hiddenSize visibleSize];
desc.X_train = patches;
desc.y_train = patches;
desc.params = theta;
desc.lambda = lambda;
desc.useSoftmax = false;
desc.spaeBeta = beta;
desc.spaeSparsity = sparsityParam;
desc.costMode = 3; % 3 => COST_RMS.

[nn_cost, nn_grad] = nn_costfunc(desc);
% [weights, costs, iters, Jcv] = nn_gradient_descent(desc);
fprintf('Plugin cost is: %f.\n',nn_cost)

assert(abs(cost-nn_cost) < 1e-10, 'Mismatch in cost values: %f != %f', cost, nn_cost);

% disp([nn_grad grad]); 

diff = norm(nn_grad-grad)/norm(nn_grad+grad);
disp(diff); % Should be small. In our implementation, these values are
assert(diff< 1e-10,'Mismatch in computed gradients')


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
