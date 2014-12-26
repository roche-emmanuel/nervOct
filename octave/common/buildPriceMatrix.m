function X =  buildPriceMatrix(data,cfg)
% This method takes as input the raw week data containing the number of minutes and the 6 colunms of data for each dataset
% and will simply convert that into a price matrix containing the open/low/high/close prices for each symbol (and the minute column)

% Check that we have the desired argments:
if exist('data', 'var') ~= 1, 
  error('must provide a dataset as input')
end

if exist('cfg', 'var') ~= 1, 
  error('must provide a configuration')
end

% Check that we have the proper number of rows in the input week dataset:
indices = cfg.minute_indices;
nrows = cfg.num_week_minutes;

% Number of minute bars to consider for each example:
nbars = cfg.num_input_bars;
npred = cfg.num_pred_bars;

% Number of symbol pairs:
nd = cfg.num_symbol_pairs;

if size(data,1)~=nrows,
  error('Invalid number of rows in week dataset: %d',size(data,1))
end

len = sum(abs(data(:,1)-indices));
if len > 0,
  error('Invalid minute indices in dataset')
end

% Prepare the resulting feature matrix:

% The number of rows to consider for building the feature matrix:
% to build each example line we need nbars previous values. So we can only start on line 'nbars' from the input dataset.
% Also we well to reserve 'npred' bars after the current line to compute the prediction label, thus, we get: (+1 because we can start from line 'nbars' and not 'nbars+1'):
n = nrows - nbars - npred + 1;

% Number of colunms per minute bar:
nc = nd*4;

% Now for each line, we need the number of minutes colunm, and for each symbol pair we will take the 4 price values (open,high,low,close), thus, the total number of colunms is:
m = 1 + nc;

X = zeros(n,m);

% From the input global dataset, we need to remove the colunms, we are not interested in (eg. tick volume and volume)
% These are the 2 last colunms accumulated for each symbol dataset.
% So we prepare an input matrix with only the colunms of interest:
% Note that we do not need the number of minutes in that input matrix:

inmat = zeros(nrows,nc);

for i=1:nd,
  inmat(:,1+4*(i-1):4*i) = data(:,2+6*(i-1):5+6*(i-1));
end

% Start buy setting the num minutes colunm in the feature matrix:
% Note that we need to offset the indices at the beginning and the end:
X(:,1) = indices(nbars:nbars+n-1,1);

% Now for each minute bar, we inject the sub matrix in the feature matrix:
X(:,2:1+nc) = inmat(nbars-i+1:nbars-i+1+n-1,:);

end

% ==> We need a valid input dataset for this method:
%!error <must provide a dataset as input> buildPriceMatrix()

% ==> We need a valid configuration for this method:
%!error <must provide a configuration> buildPriceMatrix(rand(5,4))
