clear;

prices = load('test1_eur_close_prices.txt');
returns = load('test1_eur_price_returns.txt');
signals = load('test1_eur_signals.txt');
threts = load('test1_eur_theoretical_returns.txt');
wealth = load('test1_eur_wealth.txt');
rmean = load('test1_eur_return_mean.txt');
rdev = load('test1_eur_return_dev.txt');


more off;

nmin = size(prices,1);

figure(1);
subplot(4,2,1);
plot(prices);
axis([0, nmin, min(prices(:))*0.9999, max(prices(:))*1.0001]);
subplot(4,2,2);
plot(returns);
axis([0, nmin, min(returns(:))*0.9999, max(returns(:))*1.0001]);
subplot(4,2,3);
plot(signals);
axis([0, nmin, min(signals(:))*1.1, max(signals(:))*1.1]);
subplot(4,2,4);
plot(threts);
axis([0, nmin, min(threts(:))*0.9999, max(threts(:))*1.0001]);
subplot(4,2,5);
plot(wealth);
axis([0, nmin, min(wealth(:))*0.9999, max(wealth(:))*1.0001]);
subplot(4,2,6);
plot(rmean);
axis([0, nmin, min(rmean(:))*0.9999, max(rmean(:))*1.0001]);
subplot(4,2,7);
plot(rdev);
axis([0, nmin, min(rdev(:))*0.9999, max(rdev(:))*1.0001]);

% figure(2);
% subplot(1,1,1);
% plot(eur_sharpe);
% axis([0, nmin, min(eur_sharpe(:))*1.0, max(eur_sharpe(:))*1.0]);

more on;
