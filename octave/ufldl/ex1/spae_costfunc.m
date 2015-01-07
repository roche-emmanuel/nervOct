function [cost grads] = spae_costfunc(params,desc)
	desc.params = params;
	[cost grads] = nn_costfunc(desc);
end
