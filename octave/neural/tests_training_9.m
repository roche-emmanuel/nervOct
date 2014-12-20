% ==> Draw the feature data:

%!test
%!	cfg = config();
%! 	fname = [cfg.datapath '/training_weeks_1_1_pca.mat'];
%! 	%fname = [cfg.datapath '/training_weeks_1_1.mat'];
%! 	load(fname);

%! 	% Now we can draw the evolution of the costs:
%!	figure; hold on;
%!	h = gcf();	

%	m = size(tr.X_train_raw,1);
%	plot(1:m, tr.X_train_raw(:,1+4*6), 'LineWidth', 2, 'Color','b');
%	plot(1:m, tr.X_train_raw(:,1+4*6*60), 'LineWidth', 2, 'Color','r');

%!	m = size(tr.X_train,1);
%!	plot(1:m, tr.X_train(:,1), 'LineWidth', 2, 'Color','b');
%!	plot(1:m, tr.X_train(:,2), 'LineWidth', 2, 'Color','r');


%!	legend('Jcv');
%!	title('Trade Week 1');
%!	xlabel('Number of minutes');
%!	ylabel('Value');
%!	hold off;

