function [cost grads] = spae_costfunc_dev(params,desc)
	desc.params = params;
	[cost grads] = nn_costfunc_device(desc);
end
