function validateDatasetCollection(cfg)
% will validate the date tags from a collection of forex datasets and make them synchronous.

dpath = cfg.datapath;
fpattern = cfg.symbol_name;
dname = cfg.week_dataset_name;

% more off;
% tic()

% prepare a cell array to hold the datasets:
datasets = {};
datasetNames = {};
index = 1;

% First we load all the data sets:
pname = [pwd() '/' dpath];
filelist = readdir (pname);

for ii = 1:numel(filelist)
  % skip special files . and ..
  if (regexp (filelist{ii}, '^\\.\\.?$'))
    % fprintf('Discarding file %s\n', filelist{ii})
    continue;
  end

  % chekc if the file is a text file:
  [dirname, fname, ext, vername] = fileparts (filelist{ii});
  % fprintf('Found extension %s\n',ext);
  if strcmp(ext,'.mat')==0
  	continue;
  end

  % load the file
  if (regexp (filelist{ii}, fpattern))
  	fprintf('Loading dataset %s...\n',filelist{ii})
  	
    datasets{index} = loadData([dpath '/'],fname,'dataset');
    datasetNames{index} = fname;
    index++;
  end
end

nd = size(datasets,2);
fprintf('Loaded %d datasets.\n',nd)

% check here that all input datasets have the same dimension:
for i=2:nd,
  if sum(abs(size(datasets{1}) - size(datasets{i})))~=0,
    size(datasets{1})
    size(datasets{i})
    error('Mismatch in input datasets sizes.')
  end
end

%check taht all datasets start at the same date:
for i=2:nd,
  d1 = datasets{1}(1,1:5);
  d2 = datasets{i}(1,1:5);

  if sum(abs(d1 - d2))~=0,
    d1
    d2
    error('Mismatch in input datasets start dates.')
  end
end


% Next step is to build the number of days matrix per year and month
% the matrix should covers the years from 2010 to nowadays (2014).
% We buil one colunm per year:
nyears = 5;
nmonths = 12;
numdays_mat = zeros(nmonths,nyears);

% We count the days as number of days elapsed since 1 january 2010.
doffset = floor(0.5 + datenum([2010,1,1,0,0,0]));

for y=1:nyears,
  for month=1:nmonths,
    % Compute the number of days at that date:
    dnum = floor(0.5 + datenum([2009+y, month, 1, 0, 0, 0])) - doffset;
    numdays_mat(month,y) = dnum;
  end
end

% fprintf('The numday matrix is:\n');
% numdays_mat

% number of rows in the raw datasets:
n = size(datasets{1},1);

% 1. for each dataset we should build the number of day vector:
% numdays vectors (stacked as columns in a complete matrix):
numdays = zeros(n,nd);

% now for each dataset, we accumulate the days:
for i=1:nd,
  fprintf('Computing numdays vector for %s...\n',datasetNames{i})
  data = datasets{i};
  for y=1:nyears,
    ny = 2009+y;
    for month=1:nmonths,
      numdays(:,i) += (data(:,1)==ny & data(:,2)==month) * numdays_mat(month,y);
    end
  end

  % We also need to add the number of days in the current month:
  numdays(:,i) += data(:,3);
end

% check here that the first line is the same num day value for all datasets:
if sum(abs(numdays(1,:) - numdays(1:1))) ~= 0
  numdays(1,:)
  error('Mismatch in initial start day for datasets.')
end

% fprintf('final num days values:\n')
% numdays(end,:)

% 2. For each dataset we also build the week number.

% We check what is the day of the first line of the datasets:
start_day = weekday(datenum([datasets{1}(1,1:3), 0, 0, 0]));

fprintf('Initial day index is %d\n',start_day);

% compute the week offset to apply:
% here start_day is 4 when we are on wednesday. so on next saturday, we would get 7
% since we compute weeknum as floor((dnum-offset)/7) we would then get weeknum = 1 on saturday of week 0 !
% so we need to remove 1, but then on next sunday, we get again weeknum = 1 instead of 0, so we remove 1 again.
% so the offset to **add** should be (start_day - 2) but since we want to **remove** the offset (as well as the initial day number to
% start from 0), then we have to invert the signs:
woffset = numdays(1,1) - start_day + 2;

% now for each dataset, we compute the weeknums:
% note that we can compute that for all datasets at once:
fprintf('Computing weeknum matrix for all datasets...\n')

% number of week vectors (stacked as columns):
weeknum = floor((numdays - woffset)/7);

% check here the initial week number is 0.
if sum(abs(weeknum(1,:))) ~= 0
  weeknum(1,:)
  error('Mismatch in initial week number for datasets.')
end

% fprintf('Final weeknum values:\n')
% weeknum(end,:)

% 3. For each dataset we also compute the week day number:

% We have the start_day already, so we can use that to build the day offset:

doffset = numdays(1,1) - start_day + 2;

fprintf('Computing weekdaynum matrix for all datasets...\n')

% week day vectors (stacked as matrix):
weekdaynum = mod(numdays - doffset, 7);

% 4. For each dataset we compute the total number of minutes in the week:

% To do that we can use the previously computed weekday num as well as the datasets number
% of hours and minutes:

nummins = zeros(n,nd);

for i=1:nd,
  fprintf('Computing num minutes per week for %s...\n',datasetNames{i})
  data = datasets{i};
  nummins(:,i) = weekdaynum(:,i)*(60*24) + data(:,4)*60 + data(:,5);
end

% nummins(end-100:end,:)

% 5. Now for each week available we compute the minimum and maximun number of minutes
% available in the dataset, so that we can know when is each week starting and finishing.

% Note that we do not take week 0 into account, and we only go until the latest complete week
% available for all datasets.

nweeks = min(weeknum(end,:))-1;
fprintf('We have %d complete weeks in the datasets\n', nweeks)

min_mins = zeros(nweeks,nd);
max_mins = zeros(nweeks,nd);

for i=1:nd,
  fprintf('Computing min/max min values per week for %s...\n',datasetNames{i})
  for w=1:nweeks
    mins = nummins(weeknum(:,i)==w,i);
    min_mins(w,i) = min(mins);
    max_mins(w,i) = max(mins);
  end
end

total_mins = 5*1440-1;
% min_mins
% total_mins - max_mins
% max_mins = total_mins - max_mins

% fprintf('Note that a week duration in minutes is: %d\n',total_mins)

% 6. Remove from the datasets the weeks that do not match the selection criteria:
% We want the minimal start minute to be <= 60 and we want the maximum end minutes distance to end of week to be <= 120
% first we need to merge the computation for all datasets per row:

min_limit = cfg.min_week_minutes;
max_limit = cfg.max_week_minutes;

% min_mins
% pause

min_mins = max(min_mins,[],2);
max_mins = min(max_mins,[],2);

max_mins = total_mins - max_mins;

bad_weeks = [ find(min_mins>min_limit); find(max_mins>max_limit) ];
nbad = size(bad_weeks,1);

% for each dataset, we remove the invalid weeks:
% But before doing that we need to rebuild the datasets:
% for each dataset, we should remove the date/time colunms, and 
% replace them with the weeknum and nummins colunms instead.

for i=1:nd,
  fprintf('Reconstructing dataset %s...\n',datasetNames{i})
  data = datasets{i};

  data = [weeknum(:,i) nummins(:,i) data(:,6:end)];

  % Then we remove the bad weeks from this dataset:
  % fprintf('Discarding bad weeks...\n')
  bad_vec = weeknum(:,i)==0 | weeknum(:,i)>nweeks;
  for j=1:nbad
    bad_vec = bad_vec | (weeknum(:,i)==bad_weeks(j,1));
  end
  fprintf('Discarding %d rows of bad weeks.\n',sum(bad_vec));

  data(bad_vec,:) = [];

  % We should also remove the rows where the minutes are out of range in a given week:
  nout = sum(data(:,2)<min_limit) + sum(data(:,2)>(total_mins-max_limit));
  fprintf('Discarding %d rows of out of range week minutes.\n',nout);

  % We do not discard the out of range minutes here, because
  % we have have to interpolate the value to get the proper initial or final minute !

  fprintf('Dataset %s now contains %d valid rows.\n',datasetNames{i}, size(data,1));

  datasets{i} = data;
end

% At this point we have our 6 datasets, classified from week 1 to nweeks (with the bad_weeks missing).
% Each weeks is covering min_limit to (total_mins - max_limit) minutes values inclusive.
% But we may still have some holes in them that we need to fill.

% We need to proceed separately for each week to avoid interpolating the data for the week ends.
% We will store the resulting full weeks in a cell array:

week_data = cell(2,nweeks);

fprintf('Preparing final datasets...\n')

% total number of rows per week once the data is interpolated.
nrows = total_mins - max_limit - min_limit + 1;

h = waitbar(0.0);

bad_weeks

indices = cfg.minute_indices;

for i=1:nweeks
  % check if this is a bad week, and in that case, just discard it.
  if any(bad_weeks(:,1)==i)
    continue;
  end
  
  waitbar(i/nweeks,h,sprintf('Processing week %d... (%02d%%)\n',i,floor(0.5 + 100*i/nweeks)));  
  % fprintf('Processing week %d...\n',i)

  % for each week we have to prepare a container matrix for all input datasets.
  % For each input we need 6 colunms, and we also add the number of minutes colunm.
  % We don't need the weeknum anymore since this will be retrieved from the cell index.
  % for each week we have total_mins - max_limit - min_limit + 1 rows:
  % For each dataset we will also generate a colunm indicating if the data was interpolated or not
  % those colunms will be accumulated in an additional matrix also stored in the array.
  data = zeros(nrows,1 + nd*6);

  % store the minutes indices:
  data(:,1) = indices;

  % also prepare the interpo matrix:
  interp_mat = zeros(nrows,nd);

  % process each dataset separately:
  for j=1:nd,
    % fprintf('Validating dataset %s...\n',datasetNames{j})
    dset = datasets{j}; 
    dset = dset(dset(:,1)==i,2:end); % Here we discard the weeknum col already, but we select only the rows for the desired week.

    [vdset, interp] = fillHoles(dset);

    % This is where we can finally discard the out of range minutes and still expect to get
    % the proper initial and final minute values (might have been interpolated).

    idx = vdset(:,1)<min_limit;
    vdset(idx,:) = [];
    interp(idx) = [];
    idx = vdset(:,1)>(total_mins-max_limit);
    vdset(idx,:) = [];
    interp(idx) = [];

    % check that the first/last num mins are OK:
    if vdset(1,1)~=min_limit,
      error('Invalid initial minute value %d',vdset(1,1));
    end

    if vdset(end,1)~=(total_mins - max_limit),
      error('Invalid final minute value %d',vdset(end,1));
    end


    % ensure that we have exactly the same minutes:
    len = sum(abs(vdset(:,1)-indices));
    if len>0
      vdset(end-100:end,1)
      indices(end-100:end,1)
      error('Invalid minute indices for validated dataset, len=%f',len)
    end

    % inject the dataset in the master data at the correct location:
    % size(vdset)
    % size(indices)
    data(:,2+6*(j-1):2+6*j-1) = vdset(:,2:end); % We discard the first colunm containing the num mins.

    % inject also the interpolation vector:
    interp_mat(:,j) = interp;
  end

  % Now we assign the data and interp mat to their final cell:
  week_data{1,i} = data;
  week_data{2,i} = interp_mat;
end

close(h);

fprintf('Saving final dataset %s...\n',dname)
save('-binary',[dpath '/' dname '.mat'],'week_data'); % Note that we don't need to save the bad weeks as these can easily be computed from the empty cells in the array.
save([dpath '/' dname '_order.txt'],'datasetNames'); % We save the dataset names order to know which sub index matches which symbol pair.

fprintf('Dataset collection validation done.\n',nd)

% toc()
% more on;

end

function [vdset, interp] = fillHoles(dset)
  % helper method used to perform the actual interpolation of the data when necessary.
  % compute the nmber of rows we need:
  nrows = dset(end,1)-dset(1,1) + 1;
  nc = size(dset,2);
  vdset = zeros(nrows,nc);
  interp = zeros(nrows,1);

  % initialize the first row:
  vdset(1,:) = dset(1,:);

  % Prepare the previous minute value:
  prev_min = dset(1,1);

  % next insertion index for the validated dataset:
  index = 2;

  % input num rows:
  n = size(dset,1);

  % iterate on all the other values:
  for i=2:n,
    cur_min = dset(i,1);
    if cur_min == prev_min+1,
      % Continuous data, we use that line directly
      vdset(index,:) = dset(i,:);
      index++;
      % and we increase the prev min:
      prev_min++;
    else
      prev_data = dset(i-1,:);
      cur_data = dset(i,:);

      nmins = cur_min - prev_min;
      for j = 1:nmins,
        x = j/nmins;
        prev_min++;
        idata = prev_data * (1.0 - x) + cur_data * x;
        vdset(index,:) = [prev_min idata(1,2:end)]; % normally the number of minutes should be interpolated properly too here.
        interp(index) = nmins; % save the size of the hole here.
        index++;
      end
    end
  end
end

%! block copy implementation: might improve the perfs but not working yet.

% % iterate on all the other values:
% bsize = 0;
% bstart = 2; % copy block starting on line 2;
% for i=2:n,
%   cur_min = dset(i,1);
%   if cur_min == prev_min+1,
%     % Continuous data, we use that line directly
%     % So for now we just increment the block size to copy:
%     bsize++;
%     % and we increase the prev min:
%     prev_min++;
%   else
%     % first ensure we copy the previous block if any:
%     if bsize>0
%       vdset(index:index+bsize-1,:) = dset(bstart:bstart+bsize-1,:);
%       index += bsize;
%       % reset the block:
%       bsize = 0;
%       bstart = i+1; % line i will be copied in the following interpolation process.
%     end

%     prev_data = dset(i-1,:);
%     cur_data = dset(i,:);

%     nmins = cur_min - prev_min;
%     for j = 1:nmins,
%       x = j/nmins;
%       prev_min++;
%       idata = prev_data * (1.0 - x) + cur_data * x;
%       vdset(index,:) = [prev_min idata(1,2:end)]; % normally the number of minutes should be interpolated properly too here.
%       interp(index) = nmins; % save the size of the hole here.
%       index++;
%     end
%   end
% end

% % ensure we copy the last block is any!
% if bsize>0
%   vdset(index:index+bsize-1,:) = dset(bstart:bstart+bsize-1,:);
% end  
