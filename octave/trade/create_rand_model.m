function obj = create_rand_model(desc)
	obj = create_base_model();

	obj.classname = 'rand_model';

	% assign functions:
	obj.getPrediction = @getPrediction_func;
end

% Retrieve prediction from single model:
function [obj, pred, conf] = getPrediction_func(obj)
	% fprintf('Calling rand_model.getPrediction()\n')
	pred = randi(3)-1; % return random prediction.
	conf = 1.0;	
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% tests section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%!test
%!	m = create_rand_model();
%!	[m, pred] = m.getPrediction(m);
%!	assert(pred>=0,'Invalid prediction result.')

