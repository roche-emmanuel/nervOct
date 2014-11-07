function [h] = plotLearningCurves(m_vals,jtrain_vals, jtest_vals) %, jtest_vals2
% Plot the maxmin mean_dev matrix.

% Create New Figure
figure; hold on;

h = gcf();

% Plot Examples
plot(m_vals, jtest_vals, 'LineWidth', 2, 'Color','b');
% plot(m_vals, jtest_vals2, 'LineWidth', 2, 'Color','g');
plot(m_vals, jtrain_vals, 'LineWidth', 2, 'Color','r');

% =========================================================================

% legend('Jtest','Jtest_real','Jtrain');
legend('Jtest','Jtrain');
title('Learning curves');
xlabel('Training dataset size');
ylabel('Cost function');

hold off;

end