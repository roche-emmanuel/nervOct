function list = getMissingWeeks(data)
% Method used to extract a colunm vector of missing indices from a week database.
% we assume that the database is a cell array where the number of colunms is the total number of weeks.

assert(exist('data', 'var') == 1, 'must provide a database')

% Check if we can in fact compute the bad weeks from the data set:
nweeks = size(data,2);
missing = zeros(nweeks,1);
for i=1:nweeks,
	missing(i) = isempty(data{1,i});
end

list = find(missing==1);

end
