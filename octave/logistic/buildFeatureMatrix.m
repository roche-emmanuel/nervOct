% Return the logistic feature matrix built with a given number of minutes as input.
% Note that this method will not normalize the features jus yet.
function [X, doffset] = buildFeatureMatrix(data,options)
	% Note that this method will also return the number of initial rows that should be removed.
	% This number is then used to build the label vector consistently.
	m = size(data,1);

	nm = options.time_frame;
	n_mins = options.n_mins;
	sub_last_close = options.sub_last_close;
	with_date = options.with_date;

	% In the data matrix, we have the following indexes:
	% 2014,07,15,00,00,1.36180,1.36187,1.36180,1.36186,73,72645000
	[id_year,id_month,id_day,id_hour,id_min,id_open,id_high,id_low,id_close,id_tickv,id_vol] = enum();

	% we only remove the (n_min-1) lines because we can use the current line for each example.
	doffset=n_mins-1;
	num=m-doffset-nm; % we also remove the last nm lines.

	% Keep only the columns we are interested in from the rawdata:
	mdata = data(:,[id_open,id_high,id_low,id_close]);

	nd = size(mdata,2); % number of data fields per minute.
	n = n_mins*nd; % number of minute features.

	% For each line, we have the month, the month day, the week day, the hour, the current minute, then n_mins*4 for the minute data.
	% X = zeros(num,5+n_mins*4); 

	% For each line, the hour, the current minute, then n_mins*4 for the minute data.
	% but initially we start only with the minute data.
	X = zeros(num,n); 

	% c_wd = weekday(sprintf('%02d/%02d/%4d',data(1+doffset,2),data(1+doffset,3),data(1+doffset,1)))-1; % current week day
	% c_day = data(1+doffset,3); % -1 is used to get a week day in the range 0-6

	% Reshape implementation: just a little bit slower than the sliding window implementation.
	% for i=1:num,
	% 	% for each line; we get the corresponding sub matrix of inputs from the raw data matrix:
	% 	X(i,:) = reshape(mdata(i:i+n_mins-1,:),1,n);

	% 	% if mod(i,1000)==0,
	% 	% 	fprintf('.');
	% 	% end
	% end;

	% Sliding window implementation: faster option so far. (about 0.5 secs for 7936 rows.)
	% % prepare the init vector:
	% fvec = reshape(mdata(1:n_mins-1,:)',1,n-nd);
	% % add some dummy data to initiate the sliding vector with the proper size:
	% fvec = [zeros(1,nd) fvec];

	% for i=1:num,
	% 	% Apply the vector sliding:
	% 	nvec = mdata(i+n_mins-1,:);
	% 	fvec = [fvec(1,nd+1:n) nvec];
	% 	X(i,:) = fvec - nvec(1,4); % We offset the features by the current bid price.
	% end

	% matrix sliding implementation: (about 0.018 secs for 7936 rows.)
	for i=0:n_mins-1,
		X(:,nd*i+1:nd*(i+1)) = mdata(1+i:num+i,:);
	end

	if sub_last_close,
		% Now offset everything by the last column of X:
		% And we should also drop the last column of X since it will only contain zero values.
		warning("off", "Octave:broadcast");
		X = X(:,1:n-1) - X(:,n);
		warning("on", "Octave:broadcast");
	end

	% When done we also need to add the hour and minute data to the X matrix:
	if with_date,
		X = [data(n_mins:n_mins+size(X,1)-1,[id_hour,id_min]) X];
	end
end
