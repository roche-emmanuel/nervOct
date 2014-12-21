function labels =  buildWeekLabelMatrix_AtTime(data,cfg)
% This method takes as input the raw week data containing the number of minutes and the 6 colunms of data for each dataset
% and buid a label matrix from it.

% Check that we have the desired argments:
if exist('data', 'var') ~= 1, 
  error('must provide a dataset as input')
end

if exist('cfg', 'var') ~= 1, 
  error('must provide a configuration')
end

% Then we should remove the nbars-1 top lines
% and the npred last lines from the resulting matrix:
nbars = cfg.num_input_bars;
npred = cfg.num_pred_bars;

% To start the process, we build a matrix containing the same number of rows than the input, but only one
% colunm per dataset.
nrows = size(data,1);

% number of datasets to consider:
nd = cfg.num_symbol_pairs;

% We do not need the number of minutes from the data matrix, so we remove it:

% Also, we only need the close price for each symbol so we keep only those columns:
trim_data = zeros(nrows,nd);

for i=1:nd,
	trim_data(:,i) = data(:,5+6*(i-1)); 
end

% We need to keep a reference of the close prices here:
prices = trim_data;

% Remove the history from the close prices:
prices(1:nbars-1,:) = [];

% Remove the out of range values:
prices(end-npred+1:end,:) = [];

% We should now offset the trim_data by the number of bars we when to go in the future:
trim_data(1:npred,:) = [];

% Now we should remove the history lines:
trim_data(1:nbars-1,:) = [];

% Finally we only keep the appropriate number of rows:
n = nrows - nbars - npred + 1;
trim_data = trim_data(1:n,:);

% So now we should have the same number of prices and trim_data:
assert(size(prices,1)==size(trim_data,1),'Mismatch in prices and trim_data num rows.')

% compute the difference:
trim_data = trim_data - prices;

% Now check if the difference is bigger than some threshold or not:
labels = zeros(n,nd);

labels += (trim_data >= cfg.min_gain)*1; % 1 is the id for 'buy'
labels += (trim_data <= -cfg.min_gain)*2; % 2 is the id for 'sell'

mml = max(max(labels));
assert(mml==2,'Invalid max label value: %d',mml)
end

