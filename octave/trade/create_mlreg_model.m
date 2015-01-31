% Method used to create a multi-linear-regression model.
% This model is just a collection of linear regressions that are all evaluated together 
% to produce the mean prediction.
function obj = create_mlreg_model(desc)
	if ! exist('desc', 'var')
		desc = struct();
	end

	obj = create_lreg_model(desc);

	obj.classname = 'mlreg_model';

	% assign functions:
	obj.num_instances = 5;
	obj.train = @train_func;
	obj.initTheta = @initTheta_func;
	obj.computePrediction = @computePrediction_func;

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

function pred = computePrediction_func(obj, delta)
	pred = obj.POSITION_NONE;

	% since delta can be a vector, we use the mean value of it here:
	delta = mean(delta);

	if delta > obj.min_gain
		% fprintf('Suggesting LONG pos with delta = %f\n',delta)
		pred = obj.POSITION_LONG;
	elseif delta < - obj.min_gain
		% fprintf('Suggesting SHORT pos with delta = %f\n',delta)
		pred = obj.POSITION_SHORT;
	end
end
