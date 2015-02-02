function obj = create_lreg_model(desc)
	if ! exist('desc', 'var')
		desc = struct();
	end

	obj = create_base_model(desc);

	obj.classname = 'lreg_model';

	% parameters
	obj.learning_rate = 0.00001;
	obj.conf_learning_rate = 0.0005;
	obj.gain_threshold = 1.0;
	obj.lambda = 100.0; % regularization parameter.
	obj.digest_count = 0;
	obj.use_conf_model = false;
	obj.use_return_rates = true;
	obj.use_pnorm = true;
	obj.pnorm_proj_length = 1.0; % max length of the theta vector.
	obj.pnorm = 2.0; % pnorm to use for adaptative learning rate.

	% variables:
	obj.theta = [];
	obj.conf_theta = [];
	obj.initialized = false;
	obj.max_x_norm = 0.0;
	obj.cumul_loss = [];

	% assign functions:
	obj.digest = @digest_func;
	obj.digestConfidence = @digestConfidence_func;
	obj.computePrediction = @computePrediction_func;
	obj.getPrediction = @getPrediction_func;
	obj.trainConfidence = @trainConfidence_func;
	obj.computeConfidence = @computeConfidence_func;
	obj.train = @train_func;
	obj.initTheta = @initTheta_func;
	obj.computeDelta = @computeDelta_func;

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

	obj.qnorm = pdual(obj.pnorm);
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
		obj.labels = zeros(1,2);
	end

	if obj.initialized
		% Prepare the training sample:
		% By default use only the previous value and the new label just provided.
		xx = obj.prev_x(:,2);
		yy = prev_value;

		% perform a conversion if rate of returns are requested.
		if obj.use_return_rates
			xx = ((xx ./ obj.prev_x(:,1)) - 1);
			xx(1) = 1;
			yy = ((yy / obj.labels(1,2)) - 1);
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

	obj.labels(1,1) = obj.labels(1,2);
	obj.labels(1,2) = prev_value; % This is in fact the current close price.
end

function obj = initTheta_func(obj,n)
	% fprintf('Initializing theta to size: %d.\n',n);
	obj.theta = (rand(n,1)-0.5)*2.0; % In the range [-1,1]

	% check the norm of theta:
	if(obj.use_pnorm)
		tlen = pnorm(obj.theta,obj.qnorm);
		fprintf('Initial theta qnorm: %f.\n',tlen);
		% renormalize if needed:
		if(tlen>obj.pnorm_proj_length)
			obj.theta = obj.theta*obj.pnorm_proj_length/tlen;
		end
	end
end

% function obj = train_func(obj, x, value)
% 	hx = obj.theta' * x;	
% 	obj.theta = obj.theta - obj.learning_rate * ((hx - value) * x + obj.lambda * [0.0; obj.theta(2:end)]);
% end

function obj = train_func(obj, x, value)
	% hx will be a column vector of the regression outputs.
	% We actually want to use a row vector here, so we take the transpose:
	hx = x' * obj.theta;	

	deriv = ( bsxfun(@times,(hx - value),x) + obj.lambda * [zeros(1,size(obj.theta,2)); obj.theta(2:end,:)]);

	if obj.use_pnorm
		% compute the adaptative pnorm algorithm:
		% first we get the max pnorm length:
		p = obj.pnorm;
		U = obj.pnorm_proj_length;

		max_x = max(obj.max_x_norm,pnorm(x,p));
		obj.max_x_norm = max_x;

		if size(obj.cumul_loss,2) ~= size(obj.theta,2)
			fprintf('Initializing cumul_loss with size %d.\n',size(obj.theta,2));
			obj.cumul_loss = zeros(1,size(obj.theta,2));
		end

		% add the new loss:
		obj.cumul_loss += 0.5*(hx-value).^2;

		Lt = obj.cumul_loss; % note that this is a row vector.

		kt = (obj.pnorm - 1)*max_x*max_x*U*U;

		ct = sqrt(kt) ./ (sqrt(kt+Lt) - sqrt(kt));

		nt = (ct ./ (1+ct))*(1.0/((p-1)*max_x*max_x));

		obj.theta = obj.theta - bsxfun(@times,nt,deriv);
		% size(obj.theta,2)

		tlen = pnorm(obj.theta,obj.qnorm);
		idx = tlen > U;
		obj.theta(:,idx) = bsxfun(@rdivide,obj.theta(:,idx)*U, tlen(idx));
		% if tlen > U
		% 	obj.theta = obj.theta*U ./ tlen;
		% end
	else
		obj.theta = obj.theta - obj.learning_rate * deriv;
	end
end

function obj = digestConfidence_func(obj, x, new_value)
	hx = obj.theta' * x;
	pred = obj.computePrediction(obj,hx - obj.labels(1,2));
	pred_best = obj.computePrediction(obj,new_value - obj.labels(1,2));

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

function delta = computeDelta_func(obj)
	% Prepare the training sample:
	% By default use only the previous value and the new label just provided.
	xx = obj.prev_x(:,2);
	yy = obj.labels(1,2);

	% perform a conversion if rate of returns are requested.
	if obj.use_return_rates
		xx = ((xx ./ obj.prev_x(:,1)) - 1);
		xx(1) = 1;
		%yy = ((yy / obj.labels(1,1)) - 1);
		delta = obj.theta' * xx;
	else
		delta = obj.theta' * xx - yy;
	end
end

% Retrieve prediction from single model:
function [obj, pred, conf] = getPrediction_func(obj)
	% We use the current input to build a prediction of the close price
	% Then we compare with the current close price we have,
	% if significantly higher then we go for LONG position
	% if significantly lower then we go for short position
	% Otherwise we suggest No position:
	delta = obj.computeDelta(obj);
	pred = obj.computePrediction(obj, delta);

	conf = 1.0;
	% Update the confidence value if we use the confidence model:
	if obj.use_conf_model
		conf = obj.computeConfidence(obj,xx);
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

