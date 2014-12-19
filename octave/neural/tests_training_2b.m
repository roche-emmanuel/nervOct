% ==> Should train on a network predicting positive or negative sinus

%!function np = compute_np(lsizes)
%!	np = 0;
%!	nt = numel(lsizes)-1;
%!	for i=1:nt
%!		np += (lsizes(i)+1)*lsizes(i+1);
%!	end
%!endfunction

%!test
%!	nf = 10
%!	lsizes = [nf, 32, 3]
%!	desc.lsizes = lsizes;
%!	m = 2000;
%!	desc.X_train = rand(m,nf);

%!	% build the label matrix:
%!	%xsin = sin(desc.X_train(:,1)*2*pi+desc.X_train(:,3));
%!	xsin = sin(desc.X_train(:,1)*2*pi+desc.X_train(:,3)- desc.X_train(:,8) .* desc.X_train(:,10));

%!	desc.y_train = [xsin>=0.5 xsin<=-0.5 (xsin>-0.5 & xsin<0.5)]';
%!  desc.params = rand(compute_np(lsizes),1);
%!  desc.epsilon = 0.05;
%!  desc.momentum = 0.99;
%!	desc.maxiter = 10000;
%!	desc.evalFrequency = 32;
%!	desc.miniBatchSize = 0;
%!	desc.validationWindowSize = 50;
%!	desc.X_cv = desc.X_train;
%!	desc.y_cv = desc.y_train;
%!	[weights, costs, iters] = nn_gradient_descent(desc);
%!	nn.layer_sizes = lsizes;
%!	nn.weights = weights;
%!	[y yy] = nnPredict(nn,desc.X_train);
%!	[dummy, origy] = max(desc.y_train', [], 2);
%!	origy = origy-1;
%!	y(1:20,:)
%!	%origy(1:20,:)
%!	yy(1:20,:)
%!	accuracy = 1.0 - mean(double(origy ~= y))
%! 	% Now we can draw the evolution of the costs:
%!	figure; hold on;
%!	h = gcf();	
%!	plot(iters, costs, 'LineWidth', 2, 'Color','b');
%!	legend('Jcv');
%!	title('Sin POS/CENTER/NEG Learning progress');
%!	xlabel('Number of epochs');
%!	ylabel('Cv Cost');
%!	hold off;
