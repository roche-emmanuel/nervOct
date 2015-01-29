% Test to check the mean and std value when random model for simple strategy.
function test_strategy_3(mname, mfunc)

% mfunc should be a valid model creation function
% mfunc = @create_rand_model
% mfunc = @create_lreg_model


% Initialization
% clear; close all; clc
more off;
% tic()

% Testing week range:
trange = 1:6;

cfg = config();
% fname = [cfg.datapath '/training_weeks_1_12.mat'];
% load(fname);

% Prepare the config:
cfg.num_input_bars=120;
cfg.num_pred_bars=1;
cfg.use_sparse_init = false;
cfg.use_PCA = false;
cfg.dataset_ratios = [1.00 0.0 0.0];
cfg.use_rate_of_returns = false;
cfg.discard_nmins_feature = true;
cfg.shuffle_training_data = false;

tr = nnPrepareTraining(trange,cfg);	


% Prepare the input data:
inputs = tr.X_train';

% retrieve only the columns we are interested in:
cpair = cfg.target_symbol_pair;

% Format will be High/Low/Close:
% prices = tr.prices_train;
prices = tr.prices_train(:,[1+(cpair-1)*4+2,1+(cpair-1)*4+3,1+(cpair-1)*4+4])';

ns = size(prices,2);
fprintf('Number of samples: %d.\n',ns)

% execute a serie of test to compare the statistics
% acheived with a random model and a lreg model:
ntest = 100;

final_balances = zeros(ntest,1);

tic()
for i=1:ntest
	fprintf('Running trial %d...\n',i)

	st = create_simple_strategy(struct(
		'ping_freq',0,
		'warm_up',10000));

	% Add the proper model to the strategy:
	st = st.assignModel(st, mfunc());

	[st, vals] = st.evaluate(st,inputs,prices);
	final_balances(i) = vals(end);
end
toc()

% Remove the input balance value:
final_balances = final_balances - 3000;

% Draw the balances:
figure; hold on;
h = gcf();	
plot(1:ntest, final_balances, 'LineWidth', 2, 'Color','b');
legend('Profits');
title([mname ' final Profits in EUROs']);
xlabel('Trial number');
ylabel('Profit value');
hold off;

mb = mean(final_balances);
devb = std(final_balances);

fprintf('%s profit mean: %.2f EURO.\n',mname,mb);
fprintf('%s profit standard deviation: %.2f\n',mname,devb);

more on;

end
