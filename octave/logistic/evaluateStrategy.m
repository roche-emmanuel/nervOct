% This method will be used to try and evaluate a given strategy.
% For that we need an input raw data sequence.
% And a function that will be called to actually perform the evaluation if needed.
% We also need a function to convert the raw data into features appropriately.

% The genFeatureFunc should take the rawdata matrix as only argument and generate the normalized feature matrix from it.
% it should return the feature matrix X as well as the offset into the rawdata matrix doffset.

% The evalFunc should accept a single example feature vector and return the desired action as pos_none, pos_buy or pos_sell.
% This method will return the gross_profit, the gross_lost and the number of transactions conducted.
function [gross_profit, gross_lost, num_transactions] = evaluateStrategy(options)

	data = options.test_data;

	% First we start with generating the feature matrix:
	[X, doffset] = options.genFeatureFunc(data,options);

	% normalize the features:
	X = normalizeFeatureMatrix(X,options.train_mu,options.train_sigma);

	% Add the intercept term:
	X = [ones(size(X,1), 1) X];

	% number of iterations is extracted from the feature matrix:
	m = size(X,1);

	% Prepare an enumeration with the available position labels:
	[pos_none,pos_long,pos_short] = enum(); % long position -> we just bought the security. short position -> we just sold the security

	% List of labels in the raw data matrix
	[id_year,id_month,id_day,id_hour,id_min,id_open,id_high,id_low,id_close,id_tickv,id_vol] = enum();

	% Start with no position at the beginning:
	current_pos = pos_none;

	% quantities that we won or lost so far:
	gross_profit = 0;
	gross_lost = 0;
	num_transactions = 0;

	% reference price for the latest position taken.

  % Mean spread of the bid/ask prices:
  % The mean spread comes from alpari specs at: http://www.alpari.com/why-alpari/competitive-spreads
  mean_spread = options.spread;
  max_lost = options.max_lost;

  ref_price = -1;
  stop_lost = -1;

	evalFunc = options.tradeEvalFunc;

	% Actual iteration loop:
	for i=1:m,
		current_price = data(i+doffset,id_close); % This is the close bid price at the end of the current minute.

		% When in a long position
		if current_pos == pos_long,
			current_low = data(i+doffset,id_low);

			% First check if at some point in this minute we went under the stop_lost value:
			if current_low <= stop_lost,

				if ref_price < 0 || stop_lost < 0,
					error('Invalid ref_price or stop_lost values.');
				end

				% We have to close this transaction.
				gain = stop_lost - ref_price;
				if gain > 0,
					gross_profit += gain;
				else
					gross_lost += -gain;
				end

				% Now we leave the current position:
				ref_price = -1;
				stop_lost = -1;
				current_pos = pos_none;				
			else
				% we didn't reach any stop lost yet.
				% So here we check if we are still in the "fluctuation area"
				% or if we should move to the "take highest profit" mode:
				if current_price > ref_price,
					% update the stop lost:
					new_stop = current_price - 0.5 * min(current_price - ref_price, mean_spread);

					% Only update the stop_lost if we are raising its value:
					stop_lost = max(stop_lost,new_stop);
				else
					% The price is still fluctuating between ref_price and the initial stop_lost, so we take no action here.
				end
			end
		end

		% When in a short position
		if current_pos == pos_short,
			current_high = data(i+doffset,id_high);

			% First check if at some point in this minute we went above the stop_lost value:
			if current_high >= stop_lost,

				if ref_price < 0 || stop_lost < 0,
					error('Invalid ref_price or stop_lost values.');
				end

				% We have to close this transaction.
				gain = ref_price - stop_lost;
				if gain > 0,
					gross_profit += gain;
				else
					gross_lost += -gain;
				end

				% Now we leave the current position:
				ref_price = -1;
				stop_lost = -1;
				current_pos = pos_none;				
			else
				% we didn't reach any stop lost yet.
				% So here we check if we are still in the "fluctuation area"
				% or if we should move to the "take highest profit" mode:
				if current_price < ref_price,
					% update the stop lost:
					new_stop = current_price + 0.5 * min(ref_price - current_price, mean_spread);

					% Only update the stop_lost if we are lowering its value:
					stop_lost = min(stop_lost,new_stop);
				else
					% The price is still fluctuating between ref_price and the initial stop_lost, so we take no action here.
				end
			end
		end

		% We are in no position:
		if current_pos == pos_none,
			% We can check here if we should enter a buy or sell position:
			current_pos = evalFunc(X,i);
			% fprintf('current_pos = %d\n',current_pos);

			if current_pos == pos_long,
				% We are either buying or selling, so we record the current close price:
				% When buying we have to add the mean spread to the reference price to get the ask price
				ref_price = current_price + mean_spread;

				% Stop the looses if the price goes down too much:
				stop_lost = current_price - max_lost;
				num_transactions++;

			elseif current_pos == pos_short,
				% We are either buying or selling, so we record the current close price:
				ref_price = current_price;

				% Stop the looses if the price goes high too much:
				stop_lost = current_price + max_lost;
				num_transactions++;
			else
				% not taking any position:
				ref_price = -1;
				stop_lost = -1; % reset the stop lost limit.
			end
		end
	end

end
