function [y] = buildLabelVector(data, doffset, options)

min_gain = options.min_gain;
max_lost = options.max_lost;
nm = options.time_frame;

% The time frame is used to determine how many example should be checked for each result.
% Then the min profit and max risk are used to classify that example as a donothing=0, buy=1, sell=2
% We assume here that min_gain and max_lost already take into account the spread of the security.
% eg. min_gain = spread + some_positive_value.
% eg. max_lost = spread * some_positve_ratio

m = size(data,1);

% For each line in the source data, we need to check the "nm" next lines
% To find if it is a buy/sell/donothing situation.
% Note that this cannot be computed for the last nm lines, so these are 
% Removed from the final vector result.
% TODO: here we also need to remove the initial rows that cannot be used as simples 
% when we start using history to build the features.
num=m-doffset-nm;


% In the data matrix, we have the following indexes:
% 2014,07,15,00,00,1.36180,1.36187,1.36180,1.36186,73,72645000
[id_year,id_month,id_day,id_hour,id_min,id_open,id_high,id_low,id_close,id_tickv,id_vol] = enum();


% labels to use for the y vector:
[lbl_none,lbl_buy,lbl_sell] = enum();

% fprintf('lbl_none=%d, lbl_sell=%d\n',lbl_none,lbl_sell);

% Prepare the sub data matrix:
hdata=data(1+doffset:end,id_high);
ldata=data(1+doffset:end,id_low);
bdata=data(1+doffset:end,id_close); % bid prices

high_prices = hdata(1:nm,1);
low_prices = hdata(1:nm,1);

% assert(size(high_prices,1)==nm && size(high_prices,2)==1,'Invalid high prices size');
% assert(size(low_prices,1)==nm && size(low_prices,2)==1,'Invalid low prices size');

% The initial implementation took 2.25 secs for 7936 rows.
% This implementation takes 1.05 secs for 7936 rows.

% for i=1:num,
% 	% for each line, we check the next nm bars, we need to compare each time with the ask and bit price of this line:
% 	% Note that the price we have is only the bid price (sell price).
% 	% we use the close price of the current bar as reference:
% 	bid=bdata(i,1);

% 	% Aply the window sliding:
% 	high_prices=[high_prices(2:nm,1); hdata(i+nm,1) ];
% 	low_prices=[low_prices(2:nm,1); ldata(i+nm,1) ];

% 	max_price=max(high_prices)-bid;
% 	min_price=min(low_prices)-bid;

% 	label=lbl_none;

% 	if (max_price >= min_gain) && (min_price >= -max_lost),
% 		% Here we are already in a buying condition,
% 		% So we mark the label as buy, and we stop the iteration:
% 		y(i)=lbl_buy;
% 	elseif (min_price <= -min_gain) && (max_price <= max_lost),
% 		% Here we are already in a selling condition,
% 		% So we mark the label as sell, and we stop the iteration:
% 		y(i)=lbl_sell;
% 	else
% 		y(i)=lbl_none;
% 	end
% end;

% New implementation where we build the price max and min matrices:
% This implementation only takes 0.002 secs for 7936 rows.
pmax = zeros(nm,num);
pmin = zeros(nm,num);

% Then we fill those matrices:
for i=1:nm,
	pmax(i,:) = hdata(1+i:num+i,1)';
	pmin(i,:) = ldata(1+i:num+i,1)';
end

% Now we take the min a max per column of those matrices (and we convert the result in row matrices):
% Then we substract the bid price for each row:
bid=bdata(1:num,1);
price_max = max(pmax)' - bid;
price_min = min(pmin)' - bid;

% We have 3 possible labels: buy, sell or do nothing.
% fprintf("Initializing label vector...\n")
y = ones(num,1); 

buy_vec = (price_max >= min_gain & price_min >= -max_lost);
sell_vec = (price_min <= -min_gain & price_max <= max_lost);

y += buy_vec*(lbl_buy-1) + sell_vec*(lbl_sell-1);

assert(sum(y>lbl_sell)==0,"Invalid labels found");

% Here we count the ratio of noaction/buy/sell labels:
num = size(y,1);
fprintf('Ratio of no action labels: %.2f%%\n',(sum(y==lbl_none)/num)*100);
fprintf('Ratio of buy labels: %.2f%%\n',(sum(y==lbl_buy)/num)*100);
fprintf('Ratio of sell labels: %.2f%%\n',(sum(y==lbl_sell)/num)*100);


% fprintf("Initializing done.\n")
end
