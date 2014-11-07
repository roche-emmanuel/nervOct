function [yy] = nnBuildLabelMatrix(y, num_labels=3)
% Method used to turn a label vector into a label matrix:

assert(exist('y', 'var') == 1, 'must provide a label vector')

m = size(y,1);

yy = zeros(m,3);
for i=1:m,
	yy(i,y(i)+1)=1;
end


end

% ==> Must provide the label vector:
%!error <must provide a label vector> nnBuildLabelMatrix();

% ==> Should build the matrix properly:

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

%!test
%!	num=10;
%!	for i=1:num,
%!		m = random2(10,100);
%!		nl = random2(3,5);
%!		y  = mod(1:m, nl)'; % We use 0-bazed labels.
%!		yy = nnBuildLabelMatrix(y,nl);
%!		assert(size(yy)==[m,nl],'Invalid size for label matrix.')
%!		% Now try to manually build the expected matrix:
%!		expyy = zeros(m,nl);
%!		for j=1:m,
%!			expyy(j,y(j)+1) = 1;
%!		end
%!
%!		len = sum(sum(abs(expyy-yy)));
%!		assert(len==0,'Mismacth in yy and expyy: len=%f',len);
%!	end

