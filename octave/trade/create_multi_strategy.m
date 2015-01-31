function obj = create_multi_strategy(desc)
	if ! exist('desc', 'var')
		desc = struct();
	end

	obj = create_simple_strategy(desc);

	obj.classname = 'multi_strategy';

	% obj.parameters:
	obj.num_models = 10;

	% assign functions:
	obj = rmfield(obj,'assignModel'); % There should be no assignModel function.

	obj.digest = @digest_func;
	obj.getPrediction = @getPrediction_func;
	obj.setModel = @setModel_func;

	% check if a desc argument was provided:
	% Check the arguments one by one:
	if isfield(desc, 'ping_freq')
		obj.ping_freq = desc.ping_freq;
	end
	if isfield(desc, 'warm_up')
		obj.warm_up_threshold = desc.warm_up;
	end
	if isfield(desc, 'max_lost')
		obj.max_lost = desc.max_lost;
	end
	if isfield(desc, 'min_gain')
		obj.min_gain = desc.min_gain;
	end
	if isfield(desc, 'num_models')
		obj.num_models = desc.num_models;
	end

	% parameters:
	obj.models = cell(1,obj.num_models);

end

function obj = setModel_func(obj, model, index)
	obj.models{1,index} = model;
end

% digest method to handle a single model.
function obj = digest_func(obj, x)
	% perform digest for each model:
	for i=1:obj.num_models
		m = obj.models{1,i};
		obj.models{1,i} = m.digest(m, x, obj.current_close_price);
	end
end

% Retrieve prediction from single model:
function [obj, pred, conf] = getPrediction_func(obj)
	if obj.num_models==1
		m = obj.models{1,1};
		[m, pred, conf] = m.getPrediction(m); 
		obj.models{1,1} = m;
		return;		
	end

	% Retrieve the predictions/confidences for each model
	preds = zeros(obj.num_models,2); % array will contain predictions and confidence values.

	for i=1:obj.num_models
		m = obj.models{1,i};
		[m, pred, conf] = m.getPrediction(m); 
		obj.models{1,i} = m;
		preds(i,:) = [pred, conf];
	end

	% build the count vector:
	none_idx = preds(:,1)==obj.POSITION_NONE;
	long_idx = preds(:,1)==obj.POSITION_LONG;
	short_idx = preds(:,1)==obj.POSITION_SHORT;
	count = [ sum(none_idx); sum(long_idx); sum(short_idx) ];

	% Retrieve the index of the max value:
	[max_count, idx] = max(count);

	% The final prediction we want to make is given by "idx", but idx is from 1 to 3.
	% So we need to remap:
	idx_map = [obj.POSITION_NONE, obj.POSITION_LONG, obj.POSITION_SHORT];
	pred = idx_map(idx);

	% Now compute the confidence ratio:
	other_idx = 1:3;
	other_idx(idx) = []; % Remove the index of interest.

	conf_mult = min((max_count-count(other_idx))/obj.num_models);

	conf = mean(preds(preds(:,1)==pred,2))*conf_mult;
end
