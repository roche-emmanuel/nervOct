clear;

% load('eur_prices.txt')
% load('eur_returns.txt')
% load('eur_wealth.txt')
% load('eur_signal.txt')
% load('eur_sharpe.txt')
load('eur_prices.txt')
load('eur_returns.txt')
load('eur_wealth.txt')
load('eur_signal.txt')
load('eur_sharpe.txt')


more off;

nmin = size(eur_prices,1);

% Recompute our own returns:
returns = eur_prices(2:end) - eur_prices(1:end-1);
len = sum(abs(eur_returns - returns));
assert(len<1e-12,'Mismatch in computed returns.');

%eur_wealth -= 10000.0;

figure(1);
subplot(4,1,1);
plot(eur_prices);
axis([0, nmin, min(eur_prices(:))*0.9999, max(eur_prices(:))*1.0001]);
subplot(4,1,2);
plot(eur_returns);
axis([0, nmin, min(eur_returns(:))*0.9999, max(eur_returns(:))*1.0001]);
subplot(4,1,3);
plot(eur_signal);
axis([0, nmin, min(eur_signal(:))*1.1, max(eur_signal(:))*1.1]);
subplot(4,1,4);
plot(eur_wealth);
axis([0, nmin, min(eur_wealth(:))*0.9999, max(eur_wealth(:))*1.0001]);

figure(2);
subplot(1,1,1);
plot(eur_sharpe);
axis([0, nmin, min(eur_sharpe(:))*1.0, max(eur_sharpe(:))*1.0]);


if false,

% now enter the evaluation loop:
% number of steps:
%ns = 2000;
ns = size(eur_returns,1);

trainLen = 2000;
evalLen = 200;
evalCount = 0;
ni = 10;

train_returns = zeros(trainLen,1);
eval_returns = zeros(ni,1);

% Initial theta value:
theta = ones(ni+2,1);

% returns mean and deviation:
rmean = 0.0;
rdev = 0.0;

% Training options:
options = struct('GradObj','on','Display','iter','MaxIter',1000,'HessUpdate','bfgs','GradConstr',false);

Ft_1 = 0.0;

wealth = 0.0;
profit = zeros(ns,1);
signal = zeros(ns,1);

tcost = 0.0001;

for i=1:ns
  rt = eur_returns(i);
  train_returns = push_back(train_returns,rt);
  eval_returns = push_back(eval_returns,rt);
  if(i<trainLen)
    continue;
  end

  %fprintf('On iteration %d.\n',i);

  % We have enough training examples.
  % perform training if necessary:
  if(mod(evalCount,evalLen)==0) 
    %fprintf('Performing training at step %d\n',i);
    [nrets, rmean, rdev] = featureNormalize(train_returns);

    [theta, cost, EXITFLAG] = fminlbfgs(@(t)(costFunction(nrets, train_returns, t)), ones(ni+2,1), options)
  end

  % increment eval count:
  evalCount++;

  rvec = (eval_returns-rmean)/rdev;
  params = [1; rvec; Ft_1];

  Ft = tanh(theta' * params);
  signal(i)=Ft;

  %fprintf('Predicting: Ft=%f\n',Ft);

  % Compute the Rt value:
  rt = eval_returns(end);
  Rt = Ft_1 *rt - tcost * abs(Ft - Ft_1);
  wealth += Rt;
  Ft_1 = Ft;

  profit(i) = wealth;
end

figure(2);
subplot(4,1,1);
plot(eur_prices(1:ns));
axis([0, ns, min(eur_prices(1:ns))*0.9999, max(eur_prices(1:ns))*1.0001]);
subplot(4,1,2);
plot(eur_returns(1:ns));
axis([0, ns, min(eur_returns(1:ns))*0.9999, max(eur_returns(1:ns))*1.0001]);
subplot(4,1,3);
plot(signal(1:ns));
axis([0, ns, min(signal(1:ns))*0.9999, max(signal(1:ns))*1.0001]);
subplot(4,1,4);
plot(profit(1:ns));
axis([0, ns, min(profit(1:ns))*0.9999, max(profit(1:ns))*1.0001]);

end

more on;
