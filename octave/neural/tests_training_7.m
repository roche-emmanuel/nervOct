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
%!	tr = nnPrepareTraining(1:8,cfg);
%!	nf = tr.num_features;
%!	lsizes = [nf, 512, 3]
%!	desc.lsizes = lsizes;

%!	desc.X_train = tr.X_train;
%!	desc.y_train = nnBuildLabelMatrix(tr.y_train(:,cfg.target_symbol_pair))';

%!	cfg.use_sparse_init = true;
%!	nn = nnInitNetwork(lsizes,cfg);

%!  desc.params = nn.weights;
%!  desc.epsilon = 0.0001;
%!  desc.momentum = 0.99;
%!	desc.verbose = true;
%!	desc.maxiter = 30000;
%!	desc.evalFrequency = 32;
%!	desc.miniBatchSize = 128;
%!	desc.validationWindowSize = 100;
%!	desc.minCostDecrease = 0.0;
%!	desc.learningDecay = 0.9999;
%!	desc.lambda = 0.01;

%!	desc.X_cv = tr.X_cv;
%!	desc.y_cv = nnBuildLabelMatrix(tr.y_cv(:,cfg.target_symbol_pair))';

%!	[weights, costs, iters] = nn_gradient_descent(desc);

%	% Display the weight matrices:
%	nt = numel(lsizes)-1;
%	pos = 1;
%	for i=1:nt,
%		n = lsizes(i+1);
%		m = lsizes(i)+1;
%		count = n*m;
%		mat = reshape(weights(pos:pos+count-1),n,m);
%		pos += count;
%		fprintf('Theta %d:\n',i)
%		nr = min(10,size(mat,1));
%		nc = min(10,size(mat,2));
%		mat(1:nr,1:nc)
%	end

%!	nn.weights = weights;
%!	[y yy] = nnPredict(nn,desc.X_train);
%!	[dummy, origy] = max(desc.y_train', [], 2);
%!	origy = origy-1;

%!	y(1:20,:)
%!	origy(1:20,:)
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

%!	% Now we save the generated network:
%!	nn.mu = tr.mu;
%!	nn.sigma = tr.sigma;
%!	fname = [cfg.datapath '/nn_512_3_weeks_1_4.mat'];
%!	save('-binary',fname,'nn');
