function [X_train, X_cv, X_test] = splitDataset(X, train_ratio, cv_ratio, test_ratio=0)

assert(train_ratio+test_ratio+cv_ratio<=1.0,'Too big ratios') %,train_ratio+test_ratio+cv_ratio);

m = size(X,1);

pos = 1;

% then we build the splitted datasets:
count =  floor(m*train_ratio+1/3);
X_train = X(pos:pos+count-1,:);
pos += count;

count = floor(m*cv_ratio+1/3);
X_cv = X(pos:pos+count-1,:);
pos += count;
	
count = floor(m*test_ratio+1/3);
X_test = X(pos:pos+count-1,:);
	
end

% First we need to shuffle the rows:
% idx=randperm(m);
% Xsh = X(idx,:);
% ysh = y(idx);


% ==> Should throw if too big ratios:
%!error <Too big ratios> splitDataset(rand(4,4),rand(4,5), 0.5,0.5,0.5)

% ==> Should split the datasets with the desired ratios

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

%!test
%!	num=200;
%!	for i=1:num,
%!		n = random2(1,4);
%!		m = random2(10,1000); % number of rows
%!		X = rand(m,n);
%!		train_ratio = random2(0,100);
%!		cv_ratio = random2(0,100-train_ratio);
%!		test_ratio = 100 - train_ratio - cv_ratio;
%!		[Xtrain,Xcv,Xtest] = splitDataset(X,train_ratio/100,cv_ratio/100,test_ratio/100);
%!		nb = size(Xtrain,1);
%!		assert(nb==floor(1/3 + train_ratio*m/100),'Invalid train dataset size')
%!		nb = size(Xcv,1);
%!		assert(nb==floor(1/3 + cv_ratio*m/100),'Invalid cv dataset size')
%!		nb = size(Xtest,1);
%!		assert(nb==floor(1/3 + test_ratio*m/100),'Invalid test dataset size')
%!	end
