function [weights] = nnRescaleParameters(weights,lsizes,dropouts)
% Rescale nn parameters after dropout training.

nt = numel(lsizes)-1;
offset = 0;
for i=1:nt
	count = (lsizes(i)+1)*lsizes(i+1);
	weights(offset+1:offset+count) *= dropouts(i);
	offset += count;
end

% ==> Should rescale the weights as expected:
%!test
%!	num=10;
%!	for i=1:num
%!		nl = random_int(3,7);
%!		nt = nl-1;
%!		lsizes = zeros(nl,1);
%!		for j=1:nl,
%!			lsizes(j) = random_int(3,10);
%!		end
%!	
%!		np = 0;
%!	
%!		dropouts = rand(nt,1);
%!		drop = [];
%!	
%!		for j=1:nt
%!			count = (lsizes(j)+1)*lsizes(j+1);
%!			np += count;
%!			drop = [drop; ones(count,1)*dropouts(j)];
%!		end
%!	
%!		params = rand(np,1);
%!	
%!		weights = nnRescaleParameters(params,lsizes,dropouts);
%!	
%!		pred = params .* drop;
%!	
%!		len = sum(abs(weights-pred));
%!		assert(len<1e-10,'Invalid length computation: len=%f',len);
%!	end
