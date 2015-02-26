clear;

prices = load('test1_eur_close_prices.txt');
returns = load('test1_eur_price_returns.txt');
signals = load('test1_eur_signals.txt');
threts = load('test1_eur_theoretical_returns.txt');
wealth = load('test1_eur_wealth.txt');
rmean = load('test1_eur_return_mean.txt');
rdev = load('test1_eur_return_dev.txt');
ema_sr = load('test1_eur_ema_SR.txt');
theta_norm = load('test1_eur_theta_norm.txt');


more off;

nmin = size(prices,1);

figure(1);
subplot(3,3,1);
plot(prices);
title('close prices')
axis([0, nmin, min(prices(:))*0.9999, max(prices(:))*1.0001]);
subplot(3,3,2);
plot(returns);
title('price returns')
axis([0, nmin, min(returns(:))*0.9999, max(returns(:))*1.0001]);
subplot(3,3,3);
plot(signals);
title('signals')
axis([0, nmin, min(signals(:))*1.1, max(signals(:))*1.1]);
subplot(3,3,4);
plot(threts);
title('theoretical returns')
axis([0, nmin, min(threts(:))*0.9999, max(threts(:))*1.0001]);
subplot(3,3,5);
plot(wealth);
title('wealth')
axis([0, nmin, min(wealth(:))*0.9999, max(wealth(:))*1.0001]);
subplot(3,3,6);
plot(rmean);
title('returns mean')
axis([0, nmin, min(rmean(:))*0.9999, max(rmean(:))*1.0001]);
subplot(3,3,7);
plot(rdev);
title('returns dev')
axis([0, nmin, min(rdev(:))*0.9999, max(rdev(:))*1.0001]);
subplot(3,3,8);
plot(ema_sr);
title('EMA SR')
axis([0, nmin, min(ema_sr(:))*0.9999, max(ema_sr(:))*1.0001]);
subplot(3,3,9);
plot(theta_norm);
title('Theta Norm')
axis([0, nmin, min(theta_norm(:))*0.9999, max(theta_norm(:))*1.0001]);

% figure(2);
% subplot(1,1,1);
% plot(eur_sharpe);
% axis([0, nmin, min(eur_sharpe(:))*1.0, max(eur_sharpe(:))*1.0]);

more on;
