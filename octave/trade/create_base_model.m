function obj = create_base_model(desc)
	obj.classname = 'base_model';

	% enums:
	obj.POSITION_UNKNOWN = -1;
	obj.POSITION_NONE = 0;
	obj.POSITION_LONG = 1;
	obj.POSITION_SHORT = 2;

	% assign functions:
	obj.reset = @reset_func;
	obj.digest = @digest_func;
	obj.getPrediction = @getPrediction_func;
end

function obj = reset_func(obj)
	% Not doing anything.
	% fprintf('Calling base_model.reset()\n')
end

% digest method to handle a single model.
function obj = digest_func(obj, x, prev_value)
	% fprintf('Calling base_model.digest()\n')	
end

% Retrieve prediction from single model:
function [obj, pred, conf] = getPrediction_func(obj)
	% fprintf('Calling base_model.getPrediction()\n')
	pred = obj.POSITION_UNKNOWN; % Not doing anything.
	conf = 1.0;	
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% tests section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% => Should initialize the balance properly when creating a new strategy object:

%!test
%!	m = create_base_model();
%!	[m, pred] = m.getPrediction(m);
%!	assert(pred==-1,'Invalid prediction result.')

