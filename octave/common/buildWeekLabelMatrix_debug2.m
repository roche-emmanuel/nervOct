function labels =  buildWeekLabelMatrix_debug2(data,cfg)
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
	trim_data(:,i) = data(:,5+6*(i-1)); %data(:,1)
end

% We need to keep a reference of the close prices here:
prices = trim_data;

% Remove first line for future prices:
trim_data(1,:) = [];

% Add last line:
trim_data = [ trim_data; ones(1,nd) ];

% So now we should have the same number of prices and trim_data:
assert(size(prices,1)==size(trim_data,1),'Mismatch in prices and trim_data num rows: %d != %d',size(prices,1),size(trim_data,1))

% Now take the ratio with the trimdata:
trim_data =  trim_data ./ prices;

% We should now offset the trim_data by the number of bars we when to go in the future:
trim_data(end-npred+1:end,:) = [];

% Now we should remove the history lines:
trim_data(1:nbars-1,:) = [];

% Finally we only keep the appropriate number of rows:
n = nrows - nbars - npred + 1;
assert(n==size(trim_data,1),'Mismatch in trimdata nrows: %d != %d\n',n,size(trim_data,1));
% trim_data = trim_data(1:n,:);

% Now check if the difference is bigger than some threshold or not:
labels = zeros(n,nd);

% % labels += (trim_data >= cfg.min_gain)*1; % 1 is the id for 'buy'
% % labels += (trim_data <= -cfg.min_gain)*2; % 2 is the id for 'sell'
delta = 0.00005;
labels += (trim_data >= (1.0+delta))*1; % 1 is the id for 'buy'
labels += (trim_data <= (1.0-delta))*2; % 2 is the id for 'sell'

mml = max(max(labels));
assert(mml==2,'Invalid max label value: %d',mml)
% labels = trim_data;
end

