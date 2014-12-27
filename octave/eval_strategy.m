% Initialization
clear; close all; clc
more off;
% tic()

% First we add the common location path:
pname = pwd();
% addpath(['/cygdrive/x/Station/CUDA_Toolkit-6.5/bin']); %add the binary folder.
arch=computer();
if strcmp(arch,'x86_64-w64-mingw32')==1
fprintf('Running on x64 architecture.\n')
addpath([pname '/../bin/x64']); %add the binary folder.
else
fprintf('Running on x86 architecture.\n')
addpath([pname '/../bin/x86']); %add the binary folder.
end

addpath([pname '/common']);
addpath([pname '/neural']);

cfg = config();

fname = [cfg.datapath '/training_weeks_1_12.mat'];
load(fname);
fname = [cfg.datapath '/nn_512_3_weeks_1_12.mat'];
load(fname);

% Evaluate the strategy:
cdesc.target_symbol = 6;
sid = trade_strategy('create',cdesc);

desc.type = 'nls_network';

desc.lsizes = nn.layer_sizes;
desc.params = nn.weights;
desc.mu = nn.mu';
desc.sigma = nn.sigma';

trade_strategy('add_model',sid,desc);

% Prepare the evaluation traits:
evdesc.inputs = tr.X_test_raw';

% Also inject the cv prices for the evaluation:
sym = cfg.target_symbol_pair;
evdesc.prices = (tr.prices_test(:,2+4*(sym-1):5+4*(sym-1)))';

% We also add a matrix to hold the computed balance data:
evdesc.balance = zeros(size(evdesc.inputs,2),1);

% Add a lot multiplier:
evdesc.lot_multiplier = 0.1;

% Perform evaluation:
trade_strategy('evaluate',sid,evdesc);

% once a strategy is create it should be possible to destroy it:
trade_strategy('destroy',sid)


% Draw a figure of the balance values:
figure; hold on;

h = gcf();	
x=1:size(evdesc.balance,1);
plot(x, evdesc.balance, 'LineWidth', 2, 'Color','b');
legend('Balance');
title('Balance progress');
xlabel('Number of minutes');
ylabel('Account in EURO');
hold off;

more on;
