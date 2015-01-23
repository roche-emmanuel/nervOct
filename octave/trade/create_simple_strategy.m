function obj = create_simple_strategy(desc)
	obj.classname = 'simple_strategy';

	% enums:
	obj.POSITION_NONE = 0;
	obj.POSITION_LONG = 1;
	obj.POSITION_SHORT = 2;

	% parameters:
	obj.initial_balance = 3000.0; % could be overriden by desc.

	% variables:

	% assign functions:
	obj.evaluate = @evaluate_func;
	obj.digest = @digest_func;
	obj.handleLongPosition = @handleLongPosition_func;
	obj.handleShortPosition = @handleShortPosition_func;
	obj.handleNoPosition = @handleNoPosition_func;
	obj.reset = @reset_func;

	% initialize:
	obj = obj.reset(obj,true)
end

% Note that evaluate should not update the model.
function vals = evaluate_func(obj,inputs)
	% Retrieve the number of samples:
	ns = size(inputs,2);

	% Prepare the balance value vector:
	vals = zeros(ns,1);

	% Iterate on the samples:
	for i=1:ns
		% digest the provided samples:
		obj = obj.digest(obj,inputs(:,i));

		if (obj.current_position == obj.POSITION_LONG)
			obj = obj.handleLongPosition();
		end

		if (obj.current_position == obj.POSITION_SHORT)
			obj = obj.handleShortPosition();
		end

		if (obj.current_position == obj.POSITION_NONE)
			obj = obj.handleNoPosition();
		end
			
		vals(i) = obj.current_balance;	
	end
end

function obj = reset_func(obj,resetBalance)
	if(resetBalance==true)
		fprintf('Resetting balance.\n');
		obj.current_balance = obj.initial_balance;
	end

	obj.current_position = obj.POSITION_NONE;
end

function obj = handleLongPosition_func(obj)

end

function obj = handleShortPosition_func(obj)

end

function obj = handleNoPosition_func(obj)

end

% Method used to digest a single input column:
function obj = digest_func(obj, x)

end
