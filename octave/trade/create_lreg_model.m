function obj = create_lreg_model(desc)
	if ! exist('desc', 'var')
		desc = struct();
	end

	obj = create_base_model(desc);

	obj.classname = 'lreg_model';

	% parameters
	obj.learning_rate = 0.0006;
	obj.conf_learning_rate = 0.0005;
	obj.gain_threshold = 0.3;
	obj.lambda = 0.01; % regularization parameter.
	obj.digest_count = 0;

	% variables:
	obj.theta = [];
	obj.conf_theta = [];
	obj.initialized = false;
	obj.use_conf_model = false;
	obj.use_return_rates = true;

	% assign functions:
	obj.digest = @digest_func;
	obj.digestConfidence = @digestConfidence_func;
	obj.computePrediction = @computePrediction_func;
	obj.getPrediction = @getPrediction_func;
	obj.trainConfidence = @trainConfidence_func;
	obj.computeConfidence = @computeConfidence_func;
	obj.train = @train_func;
	obj.initTheta = @initTheta_func;

	% check if a desc argument was provided:
	% Check the arguments one by one:
	if isfield(desc, 'learning_rate')
		obj.learning_rate = desc.learning_rate;
	end
	if isfield(desc, 'gain_threshold')
		obj.gain_threshold = desc.gain_threshold;
	end
	if isfield(desc, 'use_conf')
		obj.use_conf_model = desc.use_conf;
	end
	if isfield(desc, 'conf_learning_rate')
		obj.conf_learning_rate = desc.conf_learning_rate;
	end
	if isfield(desc, 'lambda')
		obj.lambda = desc.lambda;
	end
end

function obj = reset_func(obj)
	obj.initialized = false;
	obj.digest_count = 0;
end

% digest method to handle a single model.
function obj = digest_func(obj, x, prev_value)
	% Add the intercept term:
	x = [1; x];

	% first check the size of the theta vector:
	n = size(x,1);
	if size(obj.theta,1) ~= n
		obj = obj.initTheta(obj,n);

		% Here we should also init the previous_x array:
		obj.prev_x = zeros(n,2);
	end

	if obj.initialized
		% Prepare the training sample:
		% By default use only the previous value and the new label just provided.
		xx = obj.prev_x(:,2);
		yy = prev_value;

		% perform a conversion if rate of returns are requested.
		if obj.use_return_rates
			xx = (xx ./ obj.prev_x(:,1)) - 1;
			xx(1) = 1;
			yy = (yy / obj.current_value) - 1;
		end

		% If we use a confidence model then we need update it here
		% before setting the new previous features and label:
		if obj.use_conf_model
			obj = obj.digestConfidence(obj, xx, yy);
		end

		% We can train with the previous inputs and the previous value we just received:
		obj = obj.train(obj, xx, yy);
	end

	% Increment the digest count:
	obj.digest_count += 1;

	obj.initialized = (obj.digest_count>=2);

	% Since we might want to use rate of returns,
	% We need to keep the 2 previous values:
	obj.prev_x(:,1) = obj.prev_x(:,2);
	obj.prev_x(:,2) = x;

	obj.current_value = prev_value; % This is in fact the current close price.
end

function obj = initTheta_func(obj,n)
	% fprintf('Initializing theta to size: %d.\n',n);
	obj.theta = (rand(n,1)-0.5)*2.0; % In the range [-1,1]
end

function obj = train_func(obj, x, value)
	hx = obj.theta' * x;	
	obj.theta = obj.theta - obj.learning_rate * ((hx - value) * x + obj.lambda * [0.0; obj.theta(2:end)]);
end

function obj = digestConfidence_func(obj, x, new_value)
	hx = obj.theta' * x;
	pred = obj.computePrediction(obj,hx - obj.current_value);
	pred_best = obj.computePrediction(obj,new_value - obj.current_value);

	% If we got a proper prediction, then set the confidence value to one,
	% otherwise we set it to zero:
	conf_value = 0.0;
	if (pred == pred_best)
		conf_value = 1.0;
	end

	% Now we perform online training on the confidence model with the sample (prev_x, conf_value)
	obj = obj.trainConfidence(obj,obj.prev_x,conf_value);
end

function obj = trainConfidence_func(obj,x,value)
	n = size(x,1);
	if size(obj.conf_theta,1) ~= n
		obj = obj.initTheta(obj,n)
		fprintf('Initializing confidence theta to size: %d.\n',n);
		obj.conf_theta = (rand(n,1)-0.5)*2.0; % In the range [-1,1]
	end

	% Compute the current confidence that would be predicted:
	hx = obj.computeConfidence(obj,x);

	% Now compute the gradient with back propagation:
	% (We assume here a cross-entropy cost function)
	obj.conf_theta = obj.conf_theta - obj.conf_learning_rate * (hx - value) * x;
end

function conf = computeConfidence_func(obj,x)
	conf = sigmoid(obj.conf_theta' * x);
end

function pred = computePrediction_func(obj, delta)
	pred = obj.POSITION_NONE;

	if delta > obj.gain_threshold
		% fprintf('Suggesting LONG pos with delta = %f and threshold = %f\n',delta,obj.gain_threshold)
		pred = obj.POSITION_LONG;
	elseif delta < - obj.gain_threshold
		% fprintf('Suggesting SHORT pos with delta = %f and threshold = %f\n',delta,obj.gain_threshold)
		pred = obj.POSITION_SHORT;
	% else
	% 	fprintf('Suggesting NONE pos with delta = %f and threshold = %f\n',delta,obj.gain_threshold)		
	end
end

% Retrieve prediction from single model:
function [obj, pred, conf] = getPrediction_func(obj)
	% We use the current input to build a prediction of the close price
	% Then we compare with the current close price we have,
	% if significantly higher then we go for LONG position
	% if significantly lower then we go for short position
	% Otherwise we suggest No position:

	% Prepare the training sample:
	% By default use only the previous value and the new label just provided.
	xx = obj.prev_x(:,2);
	yy = prev_value;

	% perform a conversion if rate of returns are requested.
	if obj.use_return_rates
		xx = (xx ./ obj.prev_x(:,1)) - 1;
		xx(1) = 1;
		yy = (yy / obj.current_value) - 1;
	end
		
	delta = obj.theta' * obj.prev_x - obj.current_value;
	pred = obj.computePrediction(obj, delta);

	conf = 1.0;
	% Update the confidence value if we use the confidence model:
	if obj.use_conf_model
		conf = obj.computeConfidence(obj,obj.prev_x);
	end
	% fprintf('Using confidence value: %f\n',conf);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% tests section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% => Should initialize the balance properly when creating a new strategy object:

%!test
%!	m = create_lreg_model();
%!	assert(size(m.theta,1) == 0,'Invalid initial size for theta.')

