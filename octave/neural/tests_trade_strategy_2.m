
% ==> It should be possible to evaluate a strategy with a real model:
%!function np = compute_np(lsizes)
%!	np = 0;
%!	nt = numel(lsizes)-1;
%!	for i=1:nt
%!		np += (lsizes(i)+1)*lsizes(i+1);
%!	end
%!endfunction

%!test
%!	cfg = config();
%! 	fname = [cfg.datapath '/training_weeks_1_12.mat'];
%! 	load(fname);
%! 	fname = [cfg.datapath '/nn_512_3_weeks_1_12.mat'];
%! 	load(fname);
%!
%!	%nnEvaluateNetwork(tr,nn,cfg)
%!	cdesc.target_symbol = 6;
%!	sid = trade_strategy('create',cdesc);
%!
%!  desc.type = "nls_network";
%!  desc.lsizes = nn.layer_sizes;
%!	np = compute_np(desc.lsizes);
%!  desc.params = nn.weights;
%!	assert(np==numel(desc.params),'Invalid number of parameters: %d',numel(desc.params))
%!  desc.mu = nn.mu';
%!  desc.sigma = nn.sigma';
%!	trade_strategy('add_model',sid,desc);
%!
%!	% Prepare the evaluation traits:
%!	evdesc.inputs = tr.X_test_raw';
%!	
%!	% Also inject the cv prices for the evaluation:
%!	sym = cfg.target_symbol_pair;
%!	evdesc.prices = (tr.prices_test(:,2+4*(sym-1):5+4*(sym-1)))';
%!
%!	% We also add a matrix to hold the computed balance data:
%!	evdesc.balance = zeros(size(evdesc.inputs,2),1);
%!
%!	% Add a lot multiplier:
%!	evdesc.lot_multiplier = 5.0;
%!
%!	% Perform evaluation:
%!	trade_strategy('evaluate',sid,evdesc);
%!
%!	% once a strategy is create it should be possible to destroy it:
%!	trade_strategy('destroy',sid)
%! 
%! 	% Draw a figure of the balance values:
%! 	figure; hold on;
%! 	h = gcf();	
%!	x=1:size(evdesc.balance,1);
%! 	plot(x, evdesc.balance, 'LineWidth', 2, 'Color','b');
%! 	legend('Balance');
%! 	title('Balance progress');
%! 	xlabel('Number of minutes');
%! 	ylabel('Account in EURO');
%! 	hold off;

% 	figure; hold on;
% 	h = gcf();	
%	x=1:size(evdesc.balance,1);
% 	plotyy(x, evdesc.balance,x, evdesc.prices(4,:));
% 	legend('Balance','Prices');
% 	title('Balance progress');
% 	xlabel('Number of minutes');
% 	ylabel('Account in EURO');
% 	hold off;
