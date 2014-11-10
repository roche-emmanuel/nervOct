
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
%!	assert(len>0,'Invalid weight values: len=%f',len);
%!
%!	expnum = 0;
%!	for i=1:nt,
%!		expnum += (lsizes(i)+1)*lsizes(i+1);
%!	end
%!	
%!	assert(size(weights)==[expnum 1],'Mismatch in weights matrix size.')

% ==> It should be able to compute the weights for a simple XOR network.

% Helper method to predict XOR outputs:
%!function a3 = predict_xor(x1, x2, weights)
%!  a1 = sigmoid(weights(1)+x1*weights(2)+x2*weights(3));
%!  a2 = sigmoid(weights(4)+x1*weights(5)+x2*weights(6));
%!  a3 = sigmoid(weights(7)+a1*weights(8)+a2*weights(9));
%!endfunction

%!test
%!	lsizes=[2 2 1];
%!	inputs=[0 0; 1 0; 0 1; 1 1];
%!	outputs=[1; 0; 0; 1];
%!	rms_stop=0.002;
%!	[weights actual_rms] = train_bp(lsizes,inputs,outputs,rms_stop,10000);
%!
%!	% Predict the images we get from those weights:
%!	p1 = predict_xor(0,0,weights);
%!	assert(p1>0.9,'Invalid value for p1')
%!	p2 = predict_xor(1,0,weights);
%!	assert(p2<0.1,'Invalid value for p2')
%!	p3 = predict_xor(0,1,weights);
%!	assert(p3<0.1,'Invalid value for p3')
%!	p4 = predict_xor(1,1,weights);
%!	assert(p4>0.9,'Invalid value for p4')
%!	
%!	% Now compute the rms we are observing:
%!	my_rms = sqrt(((p1-1.0)*(p1-1.0) + (p2-0.0)*(p2-0.0) + (p3-0.0)*(p3-0.0) + (p4-1.0)*(p4-1.0))/4.0 ) / 2.0;
%!	assert(abs(my_rms - actual_rms)<1e-10,'Invalid result for RMS computation: %f!=%f',my_rms,actual_rms)

% ==> It should be able to simulate the sinus/cosinus functions given enough inputs:

% Helper method to predict sincos outputs:
%!function a3 = predict_sincos(x, weights)
%!	index = 1;
%!	nh1 = 30;
%!	a1 = zeros(nh1,1);
%!	x = [1 x];
%!	for i=1:nh1,
%!		a1(i) = sigmoid(x*weights(index:index+2));
%!		index += 3;
%!	end
%!	
%!	% Now we compute the activation from the second hidden layer:
%!	nh2 = 10;
%!	%index = 1;
%!	a2 = zeros(nh2,1);
%!	cur = 0;
%!	for i=1:nh2,
%!		cur = weights(index++);
%!		cur += a1'*weights(index:index+nh1-1);
%!		index += nh1;
%!		a2(i) = sigmoid(cur);
%!	end
%!	
%!	% Now compute the final layer values:
%!	nout=3;
%!	%index = 1;
%!	a3 = zeros(1,nout);
%!	cur = 0;
%!	for i=1:nout,
%!		cur = weights(index++);
%!		cur += a2'*weights(index:index+nh2-1);
%!		index += nh2;
%!		a3(i) = sigmoid(cur);
%!	end
%!endfunction

%!test
%!	m = 3000;
%!	nf = 2;
%!	nout = 3;
%!	rms_stop=0.002;
%!	lsizes = [nf 30 10 nout];
%!	inputs = zeros(m,nf);
%!	outputs = zeros(m,nout);
%!	for i=1:m,
%!		x1 = rand(1,1); %(2*i-1)/(2*m); %
%!		x2 = rand(1,1); %(2*i)/(2*m); %
%!		inputs(i,:) = [x1 x2];
%!		outputs(i,:) = abs([sin(x1) cos(x2) cos(x1)*sin(x2)]);
%!	end
%!	%inputs
%!	%outputs
%!	[weights final_rms] = train_bp(lsizes,inputs,outputs,rms_stop,1000);
%!	
%!	% Now compute the actual rms we observe:
%!	actual_rms = 0;
%!	pred = zeros(m,nout);
%!	for i=1:m,
%!		pred(i,:) = predict_sincos(inputs(i,:),weights);
%!	end
%!	
%!	dout = (outputs - pred) .* (outputs - pred);
%!	actual_rms = sqrt(sum(sum(dout))/numel(dout))/2.0;
%!	assert(abs(final_rms - actual_rms)<1e-10,'Invalid result for RMS computation on sincos : %f!=%f',final_rms,actual_rms)
