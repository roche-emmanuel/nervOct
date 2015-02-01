% Test to check the mean and std value when random model for simple strategy.
function test_strategy_3(mname, mfunc, stdesc)

if ! exist('stdesc', 'var')
	stdesc = struct();
end

% mfunc should be a valid model creation function
% mfunc = @create_rand_model
% mfunc = @create_lreg_model


% Initialization
% clear; close all; clc
more off;
% tic()

% Testing week range:
trange = 1:6;
% trange = 20:26;
% trange = 30:36;

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
ntest = 1;
% ntest = 250;
% ntest = 500;

final_balances = zeros(ntest,1);
transactions = zeros(4,ntest);

tic()
for i=1:ntest
	fprintf('Running trial %d...\n',i)

	st = create_simple_strategy(stdesc);
	% struct(
	% 	'ping_freq',0,
	% 	'warm_up',10000,
	% 	'max_lost',0.00030));

	% Add the proper model to the strategy:
	st = st.assignModel(st, mfunc());

	[st, vals] = st.evaluate(st,inputs,prices);
	final_balances(i) = vals(end);
	transactions(:,i) = st.num_transactions;
end
toc()

% Draw the balances:
figure; hold on;
h = gcf();	
plotyy(1:ns, vals, 1:ns, prices(3,:))
% plot(1:ns, vals, 'LineWidth', 2, 'Color','b');
% plot(1:ns, prices(3,:), 'LineWidth', 1, 'Color','r');
legend('Balance','Prices');
title('Balance progress in EUROs');
xlabel('Number of minutes');
ylabel('Balance value');
hold off;

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
fprintf('%s profit mean: %.2f (dev = %.2f)\n',mname,mb,devb);

% Also compute the transactions statistics:
nt = sum(transactions);
fprintf('%s num transaction mean: %.2f (dev = %.2f)\n',mname,mean(nt),std(nt));

% We are also interested in the ratio of win/lost
wr = (transactions(1,:)+transactions(3,:)) ./ max(transactions(2,:)+transactions(4,:),1);
fprintf('%s win ratio mean: %.2f (dev = %.2f)\n',mname,mean(wr),std(wr));

more on;

end
