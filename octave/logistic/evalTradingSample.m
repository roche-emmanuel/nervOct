function [label] = evalTradingSample(theta,x,options)

[label, prob] = predictLabel(theta, x);

thres = options.trading_prop_threshold;

if prob < thres,
	label = 1; % Do not trade in that case.
end

end
