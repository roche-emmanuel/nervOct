function [ vec ] = push_back( vec, val )
	n = numel(vec);
	vec(1:end-1) = vec(2:end);
	vec(end) = val;
end

