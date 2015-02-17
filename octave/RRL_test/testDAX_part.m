clear;

load('retDAX.txt')
load('DAX.txt')

more off;

M = 10; 
T = 500; % The number of time series inputs to the trader
N = 100;

 
initial_theta = ones(M+2,1); % initialize theta

X = retDAX(:); % truncate input data

Xn = featureNormalize(X);

%Ft = zeros(T+1,1);

%[Ret, sharp] = rewardFunction(X, miu, delta, Ft)

%Ft = [0; ones(M,1)]

%[Ret, sharp] = rewardFunction(X, miu, delta, Ft)

 %[ theta, U_history ] = gradientAscent(X, Ft, miu, delta, theta, alpha, num_iters, T);
 %plot(U_history)

 %  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 1000); %, 'PlotFcns', @optimplotfval);

% Evalute the cost and gradients on the initial theta position:
[cost, grad] = costFunction(Xn(1:M+T), X(1:M+T), initial_theta)

more on;
