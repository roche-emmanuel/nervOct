
% The prototype to call the function should be:
% [results] = trade_strategy(cmd,id,desc)

% ==> Should throw an error if no argument is provided:
%!error <trade_strategy: Invalid number of arguments: 0> trade_strategy()

% ==> Should throw an error if command is not a string
%!error <trade_strategy: command is not a string> trade_strategy(1,0,3)

% ==> Should throw an error if command is not known:

%!function invalid_command()
%!  desc.dummy = 10;
%!	trade_strategy('dummy',uint32(1),desc)
%!endfunction

%!error <trade_strategy: unknown command name: dummy> invalid_command()

% ==> Should throw an error if create is called without valid traits:
%!error <trade_strategy: desc should be a structure type> trade_strategy('create',3)

% ==> Should throw an error if creation traits do not contain target_symbol:

%!function invalid_creation()
%!  desc.dummy = 10;
%!	trade_strategy('create',desc)
%!endfunction

%!error <trade_strategy: target_symbol value is not defined.> invalid_creation()

% ==> Should throw an error if trying to add a model with invalid params:

%!function np = compute_np(lsizes)
%!	np = 0;
%!	nt = numel(lsizes)-1;
%!	for i=1:nt
%!		np += (lsizes(i)+1)*lsizes(i+1);
%!	end
%!endfunction

%!function no_params_model()
%!	cdesc.target_symbol = 6;
%!	sid = trade_strategy('create',cdesc);
%!  desc.type = "nls_network";
%!  desc.lsizes = [4,3,3];
%!	np = compute_np(desc.lsizes);
%!  %desc.params = rand(np,1);
%!	trade_strategy('add_model',sid,desc)
%!endfunction

%!error <trade_strategy: params value is not defined.> no_params_model()

% ==> Should throw an error if the inputs are missing for evaluation:

%!function no_prices()
%!	cdesc.target_symbol = 6;
%!	sid = trade_strategy('create',cdesc);
%!	
%!	% Prepare the evaluation traits:
%!	nf = 1 + 4*6*3;
%!	nsamples = 100;
%!	evd.inputs = rand(nf,nsamples);
%!	
%!	% Perform evaluation:
%!	trade_strategy('evaluate',sid,evd);
%!
%!	% once a strategy is create it should be possible to destroy it:
%!	trade_strategy('destroy',sid)
%!endfunction

%!error <trade_strategy: prices value is not defined.> no_prices()

% ==> It should create a valid strategy when the traits are OK:
%!test
%!	desc.target_symbol = 6;
%!	sid = trade_strategy('create',desc);
%!	assert(sid>0,'Invalid valud for Strategy id: %d',sid)
%!	% once a strategy is create it should be possible to destroy it:
%!	trade_strategy('destroy',sid)

% ==> It should be possible to evaluate a strategy:
%!test
%!	cdesc.target_symbol = 6;
%!	sid = trade_strategy('create',cdesc);
%!	
%!	% Prepare the evaluation traits:
%!	nf = 1 + 4*6*3;
%!	nsamples = 100;
%!	evdesc.inputs = rand(nf,nsamples);
%!	evdesc.prices = rand(4,nsamples);
%!	
%!	% Perform evaluation:
%!	trade_strategy('evaluate',sid,evdesc);
%!
%!	% once a strategy is create it should be possible to destroy it:
%!	trade_strategy('destroy',sid)

% ==> It should be possible to evaluate a strategy with a model loaded:
%!test
%!	cdesc.target_symbol = 6;
%!	sid = trade_strategy('create',cdesc);
%!
%!  desc.type = "nls_network";
%!  desc.lsizes = [73,10,3];
%!	np = compute_np(desc.lsizes);
%!  desc.params = rand(np,1);
%!  desc.mu = rand(73,1);
%!  desc.sigma = rand(73,1);
%!	trade_strategy('add_model',sid,desc);
%!
%!	% Prepare the evaluation traits:
%!	nf = 1 + 4*6*3;
%!	nsamples = 100;
%!	evdesc.inputs = rand(nf,nsamples)*100.0;
%!	evdesc.prices = rand(4,nsamples);
%!
%!	% Perform evaluation:
%!	trade_strategy('evaluate',sid,evdesc);
%!
%!	% once a strategy is create it should be possible to destroy it:
%!	trade_strategy('destroy',sid)

% ==> It should be possible to evaluate a strategy with a real model:
%!test
%!	cfg = config();
%! 	fname = [cfg.datapath '/training_weeks_1_1.mat'];
%! 	load(fname);
%! 	%fname = [cfg.datapath '/nn_512_128_32_3_drop_weeks_1_1.mat'];
%! 	fname = [cfg.datapath '/nn_512_3_weeks_1_8.mat'];
%! 	load(fname);
%!
%!	nnEvaluateNetwork(tr,nn,cfg)
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
%!	evdesc.inputs = tr.X_cv_raw';
%!	
%!	% Also inject the cv prices for the evaluation:
%!	sym = cfg.target_symbol_pair;
%!	evdesc.prices = (tr.prices_cv(:,2+4*(sym-1):5+4*(sym-1)))';
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
