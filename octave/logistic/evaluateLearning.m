function [ldata] = evaluateLearning(nsteps,options)
	ldata=zeros(nsteps,6);

	orig_ratio = options.train_ratio;

	% Prepare the external test dataset:
	% [X_test_base, doffset] = options.genFeatureFunc(options.test_data,options);
	% y_test = options.genLabelFunc(options.test_data,doffset,options);

	tic();
	for i=1:nsteps,
		fprintf('Starting evaluation step %d/%d...\n',i,nsteps)
		options.train_ratio = orig_ratio * i/nsteps;
		[theta, X_mu, X_sigma, J_train, J_test, train_m] = buildClassifier(options);

		% Should evaluate the trading strategy here:
		options.train_mu = X_mu;
		options.train_sigma = X_sigma;

		% fprintf('Evaluate test dataset...\n')

		% % normalize the features:
		% X_test = normalizeFeatureMatrix(X_test_base,options.train_mu,options.train_sigma);

		% % Add the intercept term:
		% X_test = [ones(size(X_test,1), 1) X_test];

		% [p, prob] = predictLabel(theta, X_test);
		% acc = mean(double(p == y_test)) * 100;
		% fprintf('External Test Accuracy: %.2f\n', acc);
		% J_test_real = 100 - acc;


		fprintf('Evaluating trading...\n')
		% This is the normal evaluation that should be used to evaluate a trading sample normally.
		options.tradeEvalFunc = @(X,i) evalTradingSample(theta,X(i,:),options);
		
		% Test method used to return perfect labels:
		% options.tradeEvalFunc = @(X,i) y_test(i);

		tic()
		% theta
		% X_mu
		% X_sigma

		[gross_profit, gross_lost, num_transactions] = evaluateStrategy(options)
		toc()

		ldata(i,1) = train_m;
		ldata(i,2) = J_train;
		ldata(i,3) = J_test;
		ldata(i,4) = gross_profit;
		ldata(i,5) = gross_lost;
		ldata(i,6) = num_transactions;
		% ldata(i,7) = J_test_real;
	end
	toc();

	fprintf('Plotting learning curves...\n')
	h = plotLearningCurves(ldata(:,1),ldata(:,2),ldata(:,3)); %,ldata(:,7)
	saveas(h,options.learning_curves_file);

	fprintf('Plotting trading curves...\n')
	h = plotTradingProfitLossCurves(ldata(:,1),ldata(:,4),ldata(:,5));
	saveas(h,options.profit_loss_curves_file);

	h = plotTradingNumTransactionCurve(ldata(:,1),ldata(:,6));
	saveas(h,options.num_transaction_curve_file);

	h = plotTradingEfficiencyCurve(ldata(:,1),ldata(:,4),ldata(:,5));
	saveas(h,options.trading_efficiency_curve_file);
end
