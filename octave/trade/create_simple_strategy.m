function obj = create_simple_strategy(desc)
	obj.classname = 'simple_strategy';

	% enums:
	obj.POSITION_UNKNOWN = -1;
	obj.POSITION_NONE = 0;
	obj.POSITION_LONG = 1;
	obj.POSITION_SHORT = 2;

	% parameters:
	obj.initial_balance = 3000.0; % could be overriden by desc.
	obj.mean_spread = 0.00008;
	obj.target_symbol_pair = 6; % 6 = EURUSD symbol. % desc.target_symbol_pair;
	obj.lot_multiplier = 0.1;
	obj.max_lost = obj.mean_spread*2.0;
	obj.min_gain = obj.mean_spread*1.2;
	obj.ping_freq = 0;
	obj.warm_up_threshold = 0; % number of iteration to wait before actually starting to trade.

	% variables:
	obj.current_position = obj.POSITION_NONE;
	obj.current_balance = 0.0;
	obj.num_transactions = 0;

	% assign functions:
	obj.evaluate = @evaluate_func;
	obj.digest = @digest_noop_func;
	obj.handleLongPosition = @handleLongPosition_func;
	obj.handleShortPosition = @handleShortPosition_func;
	obj.handleNoPosition = @handleNoPosition_func;
	obj.reset = @reset_func;
	obj.openPosition = @openPosition_func;
	obj.closePosition = @closePosition_func;
	obj.getPrediction = @getPrediction_dummy_func;
	obj.setPrices = @setPrices_func;
	obj.assignModel = @assignModel_func;

	% check if a desc argument was provided:
	if exist('desc', 'var')
		% Check the arguments one by one:
		if isfield(desc, 'ping_freq')
			obj.ping_freq = desc.ping_freq;
		end
		if isfield(desc, 'warm_up')
			obj.warm_up_threshold = desc.warm_up;
		end
	end

	% initialize:
	obj = obj.reset(obj,true);
end

% Note that evaluate should not update the model.
function [obj, vals] = evaluate_func(obj,inputs,prices)
	% Retrieve the number of samples:
	ns = size(inputs,2);

	assert(size(prices,2)==ns,'mismatch in inputs and prices nsamples.');

	% Prepare the balance value vector:
	vals = zeros(ns,1);

	% Iterate on the samples:
	for i=1:ns
		if obj.ping_freq > 0 && mod(i, obj.ping_freq)==0
			fprintf('On evaluation iteration %d...\n',i)
		end

		% Assign the current high/low/close prices:
		obj = obj.setPrices(obj, prices(:,i));

		% digest the provided samples:
		obj = obj.digest(obj,inputs(:,i));

		if i > obj.warm_up_threshold
			% The warm up phase is done, we can start actually trading.

			if (obj.current_position == obj.POSITION_LONG)
				obj = obj.handleLongPosition(obj);
			end

			if (obj.current_position == obj.POSITION_SHORT)
				obj = obj.handleShortPosition(obj);
			end

			if (obj.current_position == obj.POSITION_NONE)
				obj = obj.handleNoPosition(obj);
			end
		end
		vals(i) = obj.current_balance;	
	end
end

function obj = setPrices_func(obj,prices)
	% For each pair we receive the open/high/low/close prices:
	obj.current_close_price = prices(3); %x(obj.target_symbol_pair*4);
	obj.current_low_price = prices(2); %x(obj.target_symbol_pair*4 - 1);
	obj.current_high_price = prices(1); %x(obj.target_symbol_pair*4 - 2);
end

function obj = reset_func(obj,resetFull)
	if(resetFull==true)
		% Reset the balance as well as the evaluation statitics:
		% fprintf('Resetting balance...\n');
		obj.current_balance = obj.initial_balance;
		obj.num_transactions = 0;
	end

	obj.current_position = obj.POSITION_NONE;
end

function obj = handleLongPosition_func(obj)
  % Use current low price and stop lost values:
  % current low price should be set in the digest call:
 	% we also use the current close price and the take_profit
  if(obj.current_low_price < obj.stop_lost)
  	obj = obj.closePosition(obj);

  elseif (obj.current_close_price > obj.take_profit)
  	% Here we need to update the stop lost value:    
    new_stop = obj.current_close_price - 0.5 * min(obj.current_close_price - obj.take_profit, obj.mean_spread);

    % Only update the stop_lost if we are raising its value:	
    obj.stop_lost = max(obj.stop_lost, new_stop);
  end
end

function obj = handleShortPosition_func(obj)
  % Use current high price and stop lost values:
  % current high price should be set in the digest call:
 	% we also use the current close price and the take_profit
 	if(obj.current_high_price >  obj.stop_lost)
 		obj = obj.closePosition(obj);

 	elseif (obj.current_close_price < obj.take_profit)
  	% Here we need to update the stop lost value:    
    new_stop = obj.current_close_price + 0.5 * min(obj.take_profit - obj.current_close_price, obj.mean_spread);

    % Only update the stop_lost if we are lowering its value:	
    obj.stop_lost = min(obj.stop_lost, new_stop);
  end
end

function obj = handleNoPosition_func(obj)
	% Here we need to check what is the prediction of this strategy:
	[obj pred, confidence] = obj.getPrediction(obj);
	if(pred==obj.POSITION_NONE || pred==obj.POSITION_UNKNOWN)
		% We don't want to perform any transaction.
		return;
	end

	% Otherwise we should enter buy or sell positions:
	obj = obj.openPosition(obj,pred,confidence);
end

function obj = openPosition_func(obj, pred, confidence)
	assert(pred==obj.POSITION_LONG || pred==obj.POSITION_SHORT,'Invalid position: %d',pred);

	% Compute the number of lots to be invested:
	nlots = confidence * obj.lot_multiplier;
	nlots = floor(nlots*100.0)/100.0;
	if(nlots < 0.01)
		% We actually don't want to enter a position here:
		return;
	end

	% Otherwise we really enter the requested position:
	obj.num_lots = nlots;
	obj.current_position = pred;
	obj.transaction_price = obj.current_close_price;

	if(pred==obj.POSITION_LONG)
		obj.stop_lost = obj.current_close_price - obj.max_lost;
		obj.take_profit = obj.current_close_price + obj.min_gain;
	else
		obj.stop_lost = obj.current_close_price + obj.max_lost;
		obj.take_profit = obj.current_close_price - obj.min_gain;
	end
end

function obj = closePosition_func(obj)
	assert(obj.current_position==obj.POSITION_LONG || obj.current_position==obj.POSITION_SHORT,'Invalid position: %d',obj.current_position);
	gain = 0;
	rr = 0;
	if(obj.current_position==obj.POSITION_LONG)
		% We are about to sell since we initially bought the pair.
		% The gain will be the difference between the sell price and the buy price.
		buy_price = obj.transaction_price + obj.mean_spread;
		gain = obj.stop_lost - buy_price;
		rr = (obj.stop_lost/buy_price - 1.0);
	else
		% We are going to buy since we initially sold:
		buy_price = obj.stop_lost + obj.mean_spread;
		gain = obj.transaction_price - buy_price;
		rr = (obj.transaction_price/buy_price - 1.0);
	end

	profit = obj.num_lots * 100000 * rr;
	% fprintf('Closing position with profit of : %f euros.\n',profit);
	obj.current_balance += profit;

	% Increment the number of transactions:
	obj.num_transactions += 1;

	% Effectively close the position:
	obj.current_position = obj.POSITION_NONE;
end

% default digest method doing nothing.
function obj = digest_noop_func(obj, x, prices)

end

% digest method to handle a single model.
function obj = digest_model_func(obj, x)
	obj.model = obj.model.digest(obj.model, x, obj.current_close_price);
end

function obj = assignModel_func(obj, model)
	obj.model = model;
	obj.digest = @digest_model_func;
	obj.getPrediction = @getPrediction_model_func;
end

% Dummy prediction method to just predict to always buy:
function [obj, pred, conf] = getPrediction_dummy_func(obj)
	pred = obj.POSITION_LONG; % Always predict to buy :-)
	conf = 1.0;
end

% Retrieve prediction from single model:
function [obj, pred, conf] = getPrediction_model_func(obj)
	[obj.model, pred, conf] = obj.model.getPrediction(obj.model);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% tests section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% => Should initialize the balance properly when creating a new strategy object:

%!test
%!	st = create_simple_strategy();
%!	assert(st.current_balance == 3000.0,'Invalid initial current balance value.')

% => reset should reset the current position and [optional] the current balance:
%!test
%!	st = create_simple_strategy();
%!	st.current_position = st.POSITION_LONG;
%!	st.current_balance = 4000.0;
%!	st = st.reset(st,false);
%!	assert(st.current_position == st.POSITION_NONE,'Invalid current position')
%!	assert(st.current_balance == 4000.0,'Invalid current balance')
%!	st.current_position = st.POSITION_SHORT;
%!	st = st.reset(st,true);
%!	assert(st.current_position == st.POSITION_NONE,'Invalid current position')
%!	assert(st.current_balance == 3000.0,'Invalid current balance')

% => Run a dummy evaluation test:
%!test
%!	ns = 3000;
%!	inputs = rand(24,ns)*0.001 + 1.25;
%!	prices = rand(3,ns)*0.001 + 1.25;
%!	st = create_simple_strategy();
%!	[st, vals] = st.evaluate(st,inputs,prices);
%!	assert(size(vals,1)==ns,'Invalid number of balance values.')
%!	
%!	% Draw the balances:
%!	figure; hold on;
%!	h = gcf();	
%!	plot(1:ns, vals, 'LineWidth', 2, 'Color','b');
%!	legend('Balance');
%!	title('Balance progress in EUROs');
%!	xlabel('Number of minutes');
%!	ylabel('Balance value');
%!	hold off;

% => Run a dummy evaluation test with base model:
%!test
%!	ns = 300;
%!	inputs = rand(24,ns)*0.001 + 1.25;
%!	prices = rand(3,ns)*0.001 + 1.25;
%!	st = create_simple_strategy();
%!	m = create_base_model();
%!  st = st.assignModel(st,m);
%!	[st, vals] = st.evaluate(st,inputs,prices);
%!	assert(size(vals,1)==ns,'Invalid number of balance values.')
%!	
%!	assert(vals(end)==st.initial_balance,'Invalid finale balance value.')
%!	% Draw the balances:
%!	figure; hold on;
%!	h = gcf();	
%!	plot(1:ns, vals, 'LineWidth', 2, 'Color','b');
%!	legend('Balance');
%!	title('Balance progress in EUROs');
%!	xlabel('Number of minutes');
%!	ylabel('Balance value');
%!	hold off;


% => Run a dummy evaluation test with lreg model:
%!test
%!	ns = 300;
%!	inputs = rand(24,ns)*0.001 + 1.25;
%!	prices = rand(3,ns)*0.001 + 1.25;
%!	st = create_simple_strategy();
%!	m = create_lreg_model();
%!  st = st.assignModel(st,m);
%!	[st, vals] = st.evaluate(st,inputs,prices);
%!	assert(size(vals,1)==ns,'Invalid number of balance values.')
%!	
%!	% Draw the balances:
%!	figure; hold on;
%!	h = gcf();	
%!	plot(1:ns, vals, 'LineWidth', 2, 'Color','b');
%!	legend('Balance');
%!	title('Balance progress in EUROs');
%!	xlabel('Number of minutes');
%!	ylabel('Balance value');
%!	hold off;
