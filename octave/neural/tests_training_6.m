% ==> Should train on a network predicting positive or negative sinus

% On this training we can overfit the train data with an accuracy of 95% !
% Here we are simply using the train data as the cv data.

%!function np = compute_np(lsizes)
%!	np = 0;
%!	nt = numel(lsizes)-1;
%!	for i=1:nt
%!		np += (lsizes(i)+1)*lsizes(i+1);
%!	end
%!endfunction

%!test
%!	cfg = config();
%!	cfg.use_PCA = false;
%!	tr = nnPrepareTraining(1:1,cfg);
%!	nf = tr.num_features;
%!	lsizes = [nf, 32, 3]
%!	desc.lsizes = lsizes;
%!	desc.X_train = tr.X_train;
%!	desc.y_train = nnBuildLabelMatrix(tr.y_train(:,cfg.target_symbol_pair))';

%!	cfg.use_sparse_init = false;
%!	nn = nnInitNetwork(lsizes,cfg);

%!  desc.params = nn.weights;
%!  desc.epsilon = 0.001;
%!  desc.momentum = 0.99;
%!	desc.maxiter = 60000;
%!	desc.evalFrequency = 32;
%!	desc.miniBatchSize = 128;
%!	desc.validationWindowSize = 50;
%!	desc.X_cv = desc.X_train;
%!	desc.y_cv = desc.y_train;

%!	[weights, costs, iters] = nn_gradient_descent(desc);

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
%!	title('Trade Week Learning progress');
%!	xlabel('Number of epochs');
%!	ylabel('Cv Cost');
%!	hold off;
%!  max(costs)

