function obj = create_lreg_model(desc)
	obj.classname = 'lreg_model';

	% enums:
	obj.POSITION_UNKNOWN = -1;
	obj.POSITION_NONE = 0;
	obj.POSITION_LONG = 1;
	obj.POSITION_SHORT = 2;

	% parameters
	obj.learning_rate = 0.0005;
	obj.min_gain = 0.8;

	% variables:
	obj.theta = [];
	obj.initialized = false;

	% assign functions:
	obj.digest = @digest_func;
	obj.getPrediction = @getPrediction_func;

	% check if a desc argument was provided:
	if exist('desc', 'var')
		% Check the arguments one by one:
		if isfield(desc, 'learning_rate')
			obj.learning_rate = desc.learning_rate;
		end
		if isfield(desc, 'min_gain')
			obj.min_gain = desc.min_gain;
		end
	end	
end

function obj = reset_func(obj)
	obj.initialized = false;
end

% digest method to handle a single model.
function obj = digest_func(obj, x, prev_value)
	% Add the intercept term:
	x = [1; x];

	% first check the size of the theta vector:
	n = size(x,1);
	if size(obj.theta,1) ~= n
		fprintf('Initializing theta to size: %d.\n',n);
		obj.theta = (rand(n,1)-0.5)*2.0; % In the range [-1,1]
	end

	if obj.initialized
		% We can train with the previous inputs and the previous value we just received:
		hx = obj.theta' * obj.prev_x;
		obj.theta = obj.theta - obj.learning_rate * (hx - prev_value) * obj.prev_x;
	end

	% Now we save the new x value as the prev_x,
	% and we mark the model as initialized anyway:
	obj.initialized = true;
	obj.prev_x = x;
	obj.current_value = prev_value; % This is in fact the current close price.
end

% Retrieve prediction from single model:
function [obj, pred, conf] = getPrediction_func(obj)
	% We use the current input to build a prediction of the close price
	% Then we compare with the current close price we have,
	% if significantly higher then we go for LONG position
	% if significantly lower then we go for short position
	% Otherwise we suggest No position:

	delta = obj.theta' * obj.prev_x - obj.current_value;
	pred = obj.POSITION_NONE;

	if delta > obj.min_gain
		% fprintf('Suggesting LONG pos with delta = %f\n',delta)
		pred = obj.POSITION_LONG;
	elseif delta < - obj.min_gain
		% fprintf('Suggesting SHORT pos with delta = %f\n',delta)
		pred = obj.POSITION_SHORT;
	end

	conf = 1.0;	
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% tests section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% => Should initialize the balance properly when creating a new strategy object:

%!test
%!	m = create_lreg_model();
%!	assert(size(m.theta,1) == 0,'Invalid initial size for theta.')

