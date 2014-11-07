function [theta] = trainClassifier(X,y,nc)

n = size(X,2);

% Initialize fitting parameters
initial_theta = zeros(n, 1);

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

theta = zeros(n,nc);

warning("off", "Octave:broadcast");
for i=1:nc,
	y_test = (y==i);

	%  Run fminunc to obtain the optimal theta
	%  This function will return theta and the cost 
	[theta_vec, cost, infoVal] = fminunc(@(t)(costFunction(t, X, y_test)), initial_theta, options);

	if infoVal < 0,
		warning('fminunc returned result: %d',infoVal)
	end;

	theta(:,i) = theta_vec;
end
warning("on", "Octave:broadcast");

end
