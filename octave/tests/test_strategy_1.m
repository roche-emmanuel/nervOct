% In this training we try to first build a layer of hidden features,
% then we had a solfmax layer on top of it.

% Initialization
clear; close all; clc
more off;
% tic()


function str = rangeToString(range)
	mini = min(range);
	maxi = max(range);
	str = [int2str(mini) '_' int2str(maxi)];
end


% Testing week range:
trange = 1:6;
plsizes = [256 3];

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

tr = nnPrepareTraining(trange,cfg);	

% fname = [cfg.datapath '/training_weeks_' rangeToString(trange) '.mat'];
% save('-binary',fname,'tr');


%% ======================================================================
%  STEP 2: Prepare a strategy for evaluation:

st = create_simple_strategy();

tic()

inputs = tr.X_train_raw';
% retrieve only the columns we are interested in:
cpair = cfg.target_symbol_pair;

% Format will be High/Low/Close:
% prices = tr.prices_train;
prices = tr.prices_train(:,[1+(cpair-1)*4+2,1+(cpair-1)*4+3,1+(cpair-1)*4+4])';

ns = size(prices,2);

% Print part of that matrix:
% inputs(1:20,1:20)
% prices(:,1:20)
% fprintf('Single value: %f\n',prices(1,1));

vals = st.evaluate(st,inputs,prices);
toc()

% Draw the balances:
figure; hold on;
h = gcf();	
plot(1:ns, vals, 'LineWidth', 2, 'Color','b');
legend('Balance');
title('Balance progress in EUROs');
xlabel('Number of minutes');
ylabel('Balance value');
hold off;

fprintf('Final balance is: %.2f EURO.\n',vals(end));

more on;
