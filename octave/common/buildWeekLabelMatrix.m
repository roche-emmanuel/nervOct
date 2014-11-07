function labels =  buildWeekLabelMatrix(data,cfg)
% This method takes as input the raw week data containing the number of minutes and the 6 colunms of data for each dataset
% and buid a label matrix from it.

% Check that we have the desired argments:
if exist('data', 'var') ~= 1, 
  error('must provide a dataset as input')
end

if exist('cfg', 'var') ~= 1, 
  error('must provide a configuration')
end

% To start the process, we build a matrix containing the same number of rows than the input, but only one
% colunm per dataset.
nrows = size(data,1);

% number of datasets to consider:
nd = cfg.num_symbol_pairs;

labels = zeros(nrows,nd);

% We do not need the number of minutes from the data matrix, so we remove it:
data(:,1) = [];

for i=1:nd,
	% We build the labels separately for each dataset:
	% Note that for each dataset we have the cols in the order: open, high, low, close.
	% thus high is col 2, low is col 3 and close is col 4.
	labels(:,i) = computeLabelVector(data(:,[2+6*(i-1) 3+6*(i-1) 4+6*(i-1)]),cfg);
end

% Then we should remove the nbars-1 top lines
% and the npred last lines from the resulting matrix:
nbars = cfg.num_input_bars;
npred = cfg.num_pred_bars;

labels(1:nbars-1,:) = [];
labels(end-npred+1:end,:) = [];

end

function lbl = computeLabelVector(hlc,cfg)
	% compute a label vector from the high, low and close price values provided as argument.
	% as well as the number of prediction required.

	% Number of predictions needed:
	npred = cfg.num_pred_bars;

	% min gain and max lost values:
	min_gain = cfg.min_gain;
	max_lost = cfg.max_lost;

	nrows = size(hlc,1);

	% first we should build the high and low prices "trails" matrices so that we can compute
	% their max and min values:
	htrail = zeros(nrows,npred);
	ltrail = zeros(nrows,npred);

	for i=1:npred,
		htrail(1:end-i,i) = hlc(i+1:end,1);
		ltrail(1:end-i,i) = hlc(i+1:end,2);
	end

	% now we can compute the max/min prices for each sample:
	price_max = max(htrail,[],2) - hlc(:,3);
	price_min = min(ltrail,[],2) - hlc(:,3);

	% Prepare the resulting label vector:
	lbl = zeros(nrows,1);

	lbl += (price_max >= min_gain & price_min >= -max_lost)*1; % 1 is the id for 'buy'
	lbl += (price_min <= -min_gain & price_max <= max_lost)*2; % 2 is the id for 'sell'

	len = sum(lbl>2);
	if len>0,
		lbl
		error('Invalid label vector built.');
	end
end

% ==> We need a valid input dataset for this method:
%!error <must provide a dataset as input> buildWeekLabelMatrix()

% ==> We need a valid configuration for this method:
%!error <must provide a configuration> buildWeekLabelMatrix(rand(5,4))

% ==> check if we can detect non-buy/sel situations properly:
%!test
%!	cfg = config();
%!	nrows = 120;
%!	cfg.num_symbol_pairs = 1;
%!	data = rand(nrows,6);
%!
%!	lbl = buildWeekLabelMatrix(data,cfg);
%!	assert(size(lbl)==[56 1]);

% Helper method to build a label vector:
%!function explbl = build_label(hlc,n,nbars,npred,min_gain,max_lost)
%!	explbl = zeros(n,1);
%!	max_p = zeros(n,1);
%!	min_p = zeros(n,1);
%!	for j=1:n,
%!		% check the value we should have computed:
%!		price_max = max(hlc(nbars+j:nbars+j+npred-1,1)) - hlc(nbars+j-1,3);
%!		price_min = min(hlc(nbars+j:nbars+j+npred-1,2)) - hlc(nbars+j-1,3);
%!		%max_p(j) = price_max;
%!		%min_p(j) = price_min;
%!		if (price_max >= min_gain && price_min >= -max_lost),
%!			%	assert(lbl(j)==1,'Found label %d instead of 1',lbl(j));
%!			explbl(j,1) = 1;
%!		elseif (price_min <= -min_gain && price_max <= max_lost),
%!			%	assert(lbl(j)==2,'Found label %d instead of 2',lbl(j));
%!			explbl(j,1) = 2;
%!		%else
%!		%	assert(lbl(j)==0,'Found label %d instead of 0',lbl(j));
%!		end
%!	end	
%!endfunction

% ==> check if the reported labels are correct:
%!test
%!	num=10;
%!	for i=1:num,
%!		cfg = config();
%!		nrows = 1200;
%!		nbars = 60;
%!		npred = 5;
%!		min_gain = cfg.spread*1.2;
%!		max_lost = cfg.spread*0.5;
%!		cfg.num_symbol_pairs = 1;
%!		cfg.num_input_bars = nbars;
%!		cfg.num_pred_bars = npred;
%!		data = rand(nrows,7);
%!	
%!		lbl = buildWeekLabelMatrix(data,cfg);
%!	
%!		assert(sum(lbl)>0,'Not a single non 0 label found ???')
%!	
%!		n = size(lbl,1);
%!		explbl = build_label(data(:,[3 4 5]),n,nbars,npred,min_gain,max_lost);
%!		%min_gain
%!		%max_lost
%!		%[max_p min_p]
%!		%lbl
%!		%explbl
%!		len = sum(abs(lbl-explbl));
%!		if len>0,
%!			lbl
%!			explbl
%!			error('Mismatch in label vectors, len=%f',len);
%!		end
%!	end

% ==> check the results with multi datasets:
%!test
%!	num=10;
%!	for i=1:num,
%!		cfg = config();
%!		nrows = 1200;
%!		nbars = 60;
%!		npred = 5;
%!		min_gain = cfg.spread*1.2;
%!		max_lost = cfg.spread*0.5;
%!		cfg.num_symbol_pairs = 3;
%!		cfg.num_input_bars = nbars;
%!		cfg.num_pred_bars = npred;
%!		data = rand(nrows,1+3*6);
%!	
%!		lbl = buildWeekLabelMatrix(data,cfg);
%!		assert(size(lbl,2)==3)

%!		assert(sum(lbl)>0,'Not a single non 0 label found ???')
%!	
%!		n = size(lbl,1);
%!		for j=1:3,
%!			lblv = lbl(:,j);
%!			explbl = build_label(data(:,[3+6*(j-1) 4+6*(j-1) 5+6*(j-1)]),n,nbars,npred,min_gain,max_lost);
%!			len = sum(abs(lblv-explbl));
%!			if len>0,
%!				lblv
%!				explbl
%!				error('Mismatch in label vectors, len=%f',len);
%!			end
%!		end
%!	end

