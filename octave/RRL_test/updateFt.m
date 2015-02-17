function [ Ft ] = updateFt( X, theta, T )
%UPDATEFT Summary of this function goes here
%   Detailed explanation goes here
    M= length(theta)-2;
    
    Ft = zeros(T+1,1);
    
    % index = 1;
    for i = 2:T+1,
        xt = [1; X(i-1:i+M-2); Ft(i-1)];
       %  val = xt' * theta;
       %  if(index<=10)
       %  	theta
       %  	xt
	      %   val
	      % end
	      % index++;
        Ft(i) = tanh(xt' * theta);
    end
end

