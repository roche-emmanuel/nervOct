function [best_network, best_Jcv, all_Jcvs] = nnSelectRandomNetwork(ntrials,lsizes,training,cfg,withFig=false)
% Train a given number of NN and select the best one.

best_Jcv = Inf;
all_Jcvs = zeros(ntrials,1);

lsizes = [training.num_features lsizes];

% Enable figure display or not:
% withFig = true;

if withFig
figure; hold on;
cmap = hsv(ntrials);  %# Creates a 6-by-3 set of colors from the HSV colormap
end

for i=1:ntrials,	
	nn = nnInitNetwork(lsizes,cfg);
	
	if cfg.use_CUDA
		nn = nnTrainNetworkNERV(training,nn,cfg);
		if withFig
			plot(nn.cost_iters, nn.costs, 'LineWidth', 2, 'Color',cmap(i,:));
		end
	else
		nn = nnTrainNetwork(training,nn,cfg);
		
		ev = nnEvaluateNetwork(training,nn,cfg);
		nn.Jcv = ev.J_cv;
	end
	
	% Select this network if it is better:
	if nn.Jcv < best_Jcv
		fprintf('Selecting best NN at trial %d with cv cost %f.\n',i,nn.Jcv);
		best_network = nn;
		best_Jcv = nn.Jcv;
	end 

	% Always add the Jcv cost to the list for future stats:
	all_Jcvs(i) = nn.Jcv;
end

% Compute the mean of the Jcvs values and the deviation:
J_mean = mean(all_Jcvs);
J_dev = std(all_Jcvs);

fprintf('Mean Jcv value: %f, Deviation: %f.\n',J_mean,J_dev);

if withFig	
title('Learning progress');
xlabel('Number of epochs');
ylabel('Cv Cost');
hold off;
end

end

% ==> Should compute the cost values:
%!test
%!	cfg = config();
%!	cfg.use_PCA = false;
%!	tr = nnPrepareTraining(1:1,cfg);
%!	tr.early_stopping = true;
%!	tr.max_iterations = 0;
%!	tr.dropouts = [0.8, 0.5, 0.5, 0.5];
%!	tic();
%!	nn = nnSelectRandomNetwork(10,[128 64 32 3],tr,cfg);
%!	toc();
