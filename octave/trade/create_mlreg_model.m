% Method used to create a multi-linear-regression model.
% This model is just a collection of linear regressions that are all evaluated together 
% to produce the mean prediction.
function obj = create_mlreg_model(desc)
	if ! exist('desc', 'var')
		desc = struct();
	end

	obj = create_lreg_model(desc);

	obj.classname = 'mlreg_model';

	% parameter:
	obj.num_instances = 100;

	obj = rmfield(obj,'computePrediction'); % method not used in this model.
	obj = rmfield(obj,'trainConfidence'); % method not used in this model.
	obj = rmfield(obj,'computeConfidence'); % method not used in this model.

	% assign functions:	
	obj.train = @train_func;
	obj.initTheta = @initTheta_func;
	obj.getPrediction = @getPrediction_func;

	% check if a desc argument was provided:
	% Check the arguments one by one:
	if isfield(desc, 'num_instances')
		obj.num_instances = desc.num_instances;
	end
end

function obj = initTheta_func(obj,n)
	fprintf('Initializing multi theta to size: %dx%d.\n',n,obj.num_instances);
	obj.theta = (rand(n,obj.num_instances)-0.5)*2.0; % In the range [-1,1]
end

function obj = train_func(obj, x, value)
	% hx will be a column vector of the regression outputs.
	% We actually want to use a row vector here:
	hx = x' * obj.theta;	

	obj.theta = obj.theta - obj.learning_rate * ( bsxfun(@times,(hx - value),x) + obj.lambda * [zeros(1,size(obj.theta,2)); obj.theta(2:end,:)]);
end

% Retrieve prediction from single model:
function [obj, pred, conf] = getPrediction_func(obj)
	pred = obj.POSITION_NONE;


	delta = obj.theta' * obj.prev_x - obj.current_value;
	% % since delta can be a vector, we use the mean value of it here:
	% delta = mean(delta);

	% if delta > obj.gain_threshold
	% 	% fprintf('Suggesting LONG pos with delta = %f\n',delta)
	% 	pred = obj.POSITION_LONG;
	% elseif delta < - obj.gain_threshold
	% 	% fprintf('Suggesting SHORT pos with delta = %f\n',delta)
	% 	pred = obj.POSITION_SHORT;
	% end

	% build the count vector:
	gth = obj.gain_threshold;

	none_idx = (-gth<=delta(:) & delta(:)<=gth);
	long_idx = delta(:)>gth;
	short_idx = delta(:)<-gth;

	count = [ sum(none_idx); sum(long_idx); sum(short_idx) ]

	assert(sum(count)==obj.num_instances,'Invalid total count: %d.\n',sum(count))
	
	% Retrieve the index of the max value:
	[max_count, idx] = max(count);

	% The final prediction we want to make is given by "idx", but idx is from 1 to 3.
	% So we need to remap:
	idx_map = [obj.POSITION_NONE, obj.POSITION_LONG, obj.POSITION_SHORT];
	pred = idx_map(idx)

	% Now compute the confidence ratio:
	other_idx = 1:3;
	other_idx(idx) = []; % Remove the index of interest.

	conf = min((max_count-count(other_idx))/obj.num_instances)
end
