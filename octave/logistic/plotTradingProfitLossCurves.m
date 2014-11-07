function [h] = plotTradingProfitLossCurves(m_vals,profit_vals,lost_vals)
% Plot the maxmin mean_dev matrix.

% Create New Figure
figure; hold on;

h = gcf();

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Plot Examples
plot(m_vals, profit_vals, 'LineWidth', 2, 'Color','b');
plot(m_vals, lost_vals, 'LineWidth', 2, 'Color','r');
plot(m_vals, profit_vals-lost_vals, 'LineWidth', 2, 'Color','g');
% plot(m_vals, numt_vals, 'LineWidth', 2, 'Color','g');
% plot(m_vals, profit_vals ./ lost_vals, 'LineWidth', 2, 'Color','k');

legend('Gross Profit','Gross Loss','Net profit');
title('Gross profit & Gross Loss');
xlabel('Training dataset size');
ylabel('EURUSD per lot');

% =========================================================================

hold off;

end