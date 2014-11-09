
% The prototype to call the function should be:
% [weights] = train_bp(lsizes, inputs, outputs, rms_stop, max_iter)

% ==> Should throw an error if the first argument is missing
%!error <train_bp: should receive more than 0 arguments> train_bp()

% ==> The lsize argument should be a matrix type:
%!error <train_bp: lsize \(arg 0\) should be a matrix type> train_bp(1,2,3,4,true)

% ==> The inputs argument should be a matrix type:
%!error <train_bp: inputs \(arg 1\) should be a matrix type> train_bp([1 2 3],2,3,4,true)

% ==> Should request at least 3 layers:
%!error <train_bp: Invalid number of layers: 2> train_bp([1 3],2,3,4,true)

% ==> Should retrieve the layer size properly from col and row vectors:
%!test train_bp([2; 4; 7],rand(10,2),rand(10,7),4,10);
%!test train_bp([2 4 7],rand(10,2),rand(10,7),4,10);

% ==> Should Be able to retrieve input data by row directly from the matrix:
%!test train_bp([33; 4; 10],rand(100,33),rand(100,10),4,10);

% ==> Should return the proper number of weights:

%!test
%!	lsizes = [33 4 10];
%!	nt=numel(lsizes)-1;
%!	rms_stop=0.002;
%!	max_iter=100;
%!	inputs=rand(100,33);
%!	outputs=rand(100,10);
%!	
%!	weights = train_bp(lsizes,inputs,outputs,rms_stop,max_iter);
%!	
%!	% For now, all the weights should be set to 0:
%!	len = sum(abs(weights));
%!	assert(len==0,'Invalid weight values: len=%f',len);
%!
%!	expnum = 0;
%!	for i=1:nt,
%!		expnum += (lsizes(i)+1)*lsizes(i+1);
%!	end
%!	
%!	assert(size(weights)==[expnum 1],'Mismatch in weights matrix size.')
