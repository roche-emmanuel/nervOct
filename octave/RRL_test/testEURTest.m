clear; clc;

load('test_wealth.txt')
load('test_signal.txt')

more off;

nmin = size(test_wealth,1);

figure(1);
subplot(2,1,1);
plot(test_signal);
axis([0, nmin, min(test_signal(:))*1.1, max(test_signal(:))*1.1]);
subplot(2,1,2);
plot(test_wealth);
axis([0, nmin, min(test_wealth(:))*0.9999, max(test_wealth(:))*1.0001]);

more on;
