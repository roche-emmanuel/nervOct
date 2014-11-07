function [result, interpvec] = validateDataset(dataset,dbg=false,maxgap=1440)
% This method is used to check that an input dataset is consistent for forex trading.
% It will mainly ensure that there are no hole left in a dataset,
% And in case holes are found, they will be filled with interpolated data
% This function will also ensure that the resulting dataset will keep the initial size.

assert(exist('dataset', 'var') == 1, 'must a dataset as input')

tic()

% get the initial number of rows:
n = size(dataset,1);

% Show progress if n is large
progress = n>100000;

% ensure that we have the year/month/data/hour/min colunms (eg. at least 5 colunms)
m = size(dataset,2);
assert(m>=5, 'not enough colunms in input dataset')

% We check the validity of each line except the first one:
result = zeros(n,m);
result(1,:) = dataset(1,:);

% interpolation vector used to store the ise of the interpolation range (minus 1).
interpvec = zeros(n,1);

% duration of 1 minute in day nums:
dt = 1/(60*24);

% prepare the next insertion index:
index=2;

% init the number of holes found:
nholes = 0;

count = 0;
biggest = 0;
biggestVec = [];
ninterp = 0;

% As long as we have the same day base, there is no need to recompute the datenum,
% and we can just accumulate the number of minutes:
day_vec = result(1,1:3);
prev_day_vec = day_vec;
dbase = datenum([day_vec, 0, 0, 0]);
prev_dbase = dbase;

% prepare the vector of number of minutes:
mindt = (dataset(:,4)*60.0 + dataset(:,5))*dt;

% Now compute the current daynum:
prev_dnum = dbase + mindt(1);

for i=2:n,
  % check if we need to recompute the dbase value
  % eg. if we are on another day number:
  if sum((day_vec-dataset(i,1:3))) ~= 0
    % Keep a copy of the previous values in case we need to recompute
    % the date vec (if there is an hole in the data).
    prev_day_vec = day_vec;
    prev_dbase = dbase;

    % We need to recompute the date base:
    day_vec = dataset(i,1:3);
    dbase = datenum([day_vec, 0, 0, 0]);
  end

  % Compute the current day num:
  dnum = dbase + mindt(i);

  % Then we need to count how many minutes were "jumped" in this hole.
  nmins = floor(0.5 + (dnum-prev_dnum)/dt);
  
  if nmins==0
    error('Found duplicated data on line %d',i);
  end

  if nmins==1,
    % there is no jump here, so we can use the values as is:
    result(index,:) = dataset(i,:);

    % update the prev_dnum value:
    prev_dnum = dnum;

    % increment the insertion index:
    index++;
  else
    % There is a mismatch with the previous line.
    % So this mean we should fill the holes.

    % first increment the number of holes:
    nholes++;

    % We have to add a line for each minute missed:
    % So first we need to prepare the data to be interpolated:
    prev_data = dataset(i-1,6:end);
    cur_data = dataset(i,6:end);

    if dbg
      % vec = datevec(prev_dnum);
      emins = (prev_dnum - prev_dbase)/dt;
      hval = floor(emins/60.0);
      vec = [prev_day_vec, hval, emins - hval];

      if !progress
        fprintf('Found hole of %d minutes after %d/%d/%d %d:%d\n',nmins,vec(1),vec(2),vec(3),vec(4),vec(5));
      end
      if nmins > biggest,
        biggest = nmins;
        biggestVec = vec;
      end
    end

    for j=1:nmins,
      % Generate the interpolated data:
      x = j/nmins;
      data = prev_data * (1.0-x) + cur_data * x;

      % Prepare the corresponding date vector:
      prev_dnum += dt;
      vec = datevec(prev_dnum);

      % fix the values for the number of minutes taking
      % the number of seconds into account:
      if vec(6)>=30,
        vec(5) += 1;
        vec(6) = 0;
        vec = datevec(datenum(vec));
      end
      if vec(6)>=30
        error('Invalid number of seconds: %d',vec(6);
      end

      % now inject that line in the results:
      result(index,:) = [vec(1,1:5) data];
      interpvec(index) = nmins;

      % increment the number of interpolations:
      ninterp++;

      % increment the insertino index:
      index++;
      if index>n,
        break;
      end
    end
  end

  if index>n,
    break;
  end

  count++;
  if progress && ((count/n) > 0.01),
    fprintf('*');
    count = 0;
  end
end

if progress
  fprintf('\n');
end

if dbg
  nmins = biggest;
  vec = biggestVec;
  fprintf('Found %d holes in dataset. (Total of %d minutes generated).\nBiggest hole in dataset is %d minutes after %d/%d/%d %d:%d\n',nholes,ninterp,nmins,vec(1),vec(2),vec(3),vec(4),vec(5));
end

toc()

end


% ==> We need a valid input for this method:
%!error <must a dataset as input> validateDataset()

% ==> We need to have at least 5 colunms in the input matrix:
%!error <not enough colunms in input dataset> validateDataset(rand(5,4))

% ==> Check that we can do proper calculations on date.
% we start with a date vector [y, m, d, h, m, s]
% Then we can convert that to a number of days (with fraction) since Jan 1, 0000 with datenum
% Then we increment this value by 1 minute of time, since we have 60*24 minutes per day
% this meanw we should increase the value by 1/(60*24)
% then we convert back to to a date vector with datevec,
% we we expect to see an increase of one minute only.

%!test
%!  v = [2010, 1, 31, 23, 59, 00];
%!  dt = 1/(60*24);
%!  dnum = datenum(v) + dt;
%!  v2 = datevec(dnum);
%!  assert(v2(1)==2010);
%!  assert(v2(2)==2);
%!  assert(v2(3)==1);
%!  assert(v2(4)==0);
%!  assert(v2(5)==0);
%!  assert(v2(6)==0);

% Since octave handles out of range dates, we can automate the computation for random dates:

% Helper method to return a random value between 1 and max
%!function val = random(max)
%!  val = ceil(rand(1,1)*max);
%!endfunction

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

%!test
%!  num = 1000;
%!  for i=1:num
%!    mm = ceil(rand(1,1)*100);
%!    val = random(mm);
%!    assert(1<=val && val<=mm,sprintf('Invalid random value %d, for max value %d',val,mm));
%!  end

%!test
%!  num = 1000;
%!  for i=1:num
%!    mm = ceil(rand(1,1)*100);
%!    mm2 = ceil(rand(1,1)*100);
%!    mini = min(mm,mm2);
%!    maxi = max(mm,mm2);
%!    val = random2(mini,maxi);
%!    assert(mini<=val && val<=maxi,sprintf('Invalid random value %d, for min/max value %d/%d',val,mini,maxi));
%!  end

%test
%!  num = 100;
%!  for i=1:num,
%!    % prepare a random initial date vector: 
%!    v1 = [random(2050),random(12),random(31),random(24),random(59),random(59)];
%!    % compute a random number of minutes:
%!    dmin = random(100);
%!    dnum = datenum(v1);
%!    v1(5) += dmin;
%!    dnum2 = datenum(v1);
%!    expnum = dnum+dmin/(60*24);
%!    assert(abs(expnum-dnum2)<1e-6,sprintf('Mismatch in datenum values %f != %f',expnum,dnum2));
%!  end

% ==> check that datenum can be constructed when omittinf the number of seconds:

%!test
%!  num = 20;
%!  for i=1:num,
%!    v1 = [random(2050),random(12),random(31),random(24),random(59),0];
%!    v2 = v1(1,1:5);
%!    dnum1=datenum(v1);
%!    dnum2=datenum(v2);
%!    assert(dnum1==dnum2,sprintf('Mismatch in datenums when ommitting numsecs %f != %f',dnum1,dnum2));
%!  end

% ==> Here we should ensure that a function can change its argument matrices in place:
% => It turns out that this test is failing, which means that we need to return the dataset when we modify it.

%!function mat = make_ones(mat)
%!  n = size(mat,1);
%!  m = size(mat,2);
%!  for r=1:n,
%!    for c=1:m,
%!      mat(r,c) = 1;
%!    end
%!  end
%!endfunction

%!test
%!  num = 5;
%!  for i=1:num,
%!    m = rand(3,3);
%!    m = make_ones(m);
%!    for r=1:3,
%!      for c=1:3,
%!        assert(m(r,c)==1,'Expect value of 1 here.')
%!      end
%!    end
%!  end

% ==> If there is only one line, we just return that line:

%!test
%!  num=10;
%!  for i=1:num,
%!    m = rand(1,6);
%!    m2 = validateDataset(m);
%!    assert(size(m)==size(m2),'Invalid resulting one line dataset')
%!  end

% ==> So now we want to read each line of the input dataset and create a dnum value for it
% then we compare this value with the previous value and check that we get a difference of 1/(60*24) (which is one minute duration)
% if this is not the case, it means we have a jump.
% in case we have a jump we compute the number of minutes that were "jumped". This will tell us the number of lines that should be inserted.
% First we can test when there is no hole in the input dataset:

% Helper method used to generate a random dataset with no hole in it:
% the n argument is the number of lines
% the m argument is the number of additional colunms after the 5 date colunms.
%!function mat = gen_dataset(n,m)
%!  mat = zeros(n,m+5);
%!  % prepare an random initial date:
%!  vbase = [random(2050),random(12),random(31),random(24),random(59)];
%!  dt = 1/(60*24);
%!
%!  % Populate the lines of the matrice:
%!  count = 0;
%!  progress = n>100000;
%!  for i=1:n,
%!    vbase(5) += 1;
%!    % convert that value to a dnum:
%!    dnum = datenum(vbase);
%!    % convert back to a proper date vec:
%!    vec = datevec(dnum);
%!    if vec(6)>=30,
%!      % We need to recompute the date vector:
%!      vec(5) += 1;
%!      vec(6) = 0;
%!      vec = datevec(datenum(vec));
%!    end
%!    assert(vec(6)<30,'Invalid number of seconds');
%!    mat(i,:) = [vec(1,1:5), rand(1,m)];
%!    count++;
%!    if progress && ((count/n)>0.01),
%!      fprintf('*');
%!      count = 0;
%!    end
%!  end
%!  if progress,
%!    fprintf('\n');
%!  end
%!endfunction

% Simple test to ensure visually that the gen_dataset method is working:
%test
%!  num=10;
%!  for i=1:num,
%!    gen_dataset(10,1)
%!  end

%!test
%!  num = 10;
%!  for i=1:num,
%!    n =  random2(30,100);
%!    mat = gen_dataset(n,6);
%!  
%!    % now try to validate that dataset:
%!    [vmat, interp] = validateDataset(mat);
%!    nholes = sum(double(interp > 0));
%!    assert(nholes==0,sprintf('Found %d holes in consistent dataset test.',nholes))
%!  
%!    % compare the size of the matrices:
%!    assert(size(mat)==size(vmat),'Mismatch in matrix sizes')
%!  
%!    % compare the values in the matrices:
%!    for r=1:n,
%!      for c=1:6,
%!        assert(mat(r,c)==vmat(r,c),'Mismatch in matrix values')
%!      end
%!    end
%!  end

% ==> check if we can define functions inside functions
% => It seems that we cannot define nested function in test code.
% Yet, this seems to work properly when implemented in regular function files (both nested functions and sub functions)

%!function mat = func_parent(n,m)
%!  mat = n+m;  
%!endfunction

%test
%  num=5;
%  for i=1:num,
%    a = rand(1,1);
%    b = rand(1,1);
%    c = a+b;
%    d = func_parent(a,b);
%    assert(c==d,sprintf('Mismatch when calling func_parent: %f != %f',c,d));
%  end

% ==> Now we when to test with matrices containing holes
% So we use our gen_dataset method to generate complete matrices, and then we manually create holes in those matrices
% before trying to validate them.

%!test
%!  num = 10;
%!  for i=1:num,
%!    n = random2(100,200);
%!    mat = gen_dataset(n,0);
%!    data = floor(rand(n,1)*10 + 0.5);
%!    matd = [mat data];
%!  
%!    % we should at least leave the beginning and end values of the matrix to test:
%!    nh = random2(1,n-2);
%!  
%!    % now select the row indices that we want to remove:
%!    % note that the resulting indices are in the range [1,n-2] so
%!    % we need to add 1 to get into the range [2,n-2], and thus avoid removing the beginning and end of matrix.
%!    idx = randperm(n-2,nh)+1;
%!  
%!    % Remove the desired lines from the matrix:
%!    nmat = matd;
%!    nmat(idx,:) = [];
%!  
%!    n1 = size(nmat,1);
%!    n2 = size(matd,1);
%!    assert(n1==(n2-nh),sprintf('Didnt expect that size for new mat: %d != %d - %d',n1,n2,nh));
%!    [vmat, interp] = validateDataset(nmat,true);
%!    % matd
%!    % nmat
%!    % vmat
%!    nholes = sum(double(interp > 0));
%!    assert(size(nmat)==size(vmat),'invalid size for validated dataset ?')
%!    assert(nholes>0,'Found no hole in inconsistent dataset')
%!  
%!    % Ensure that that date part of the matrix is OK:
%!    % To check that we substract from the original matrix,
%!    % then take the abs and sum everything => expecting to get 0.
%!  
%!    difmat = abs(vmat(:,1:5) - matd(1:size(vmat,1),1:5));
%!    len = sum(sum(difmat));
%!    assert(len==0,sprintf('Invalid length different for date part: %f',len));
%!  
%!    % ensure that the interpolated data is OK:
%!    % check which line should be interpolated:
%!    interp = zeros(n,1);
%!    interp(idx) = 1;
%!    % interp
%!    % idx
%!    % Now we build the vector of interpolated data:
%!    intdata = zeros(n,1);
%!    for j=1:n,
%!      if interp(j),
%!        % The value shoud be interpolated.
%!        % We need to find the previous and the next indices of zeros:
%!        prev_index = j;
%!        next_index = j;
%!        while true,
%!          prev_index--;
%!          if(interp(prev_index)==0)
%!            break;
%!          end
%!        end
%!        while true,
%!          next_index++;
%!          if(interp(next_index)==0)
%!            break;
%!          end
%!        end
%!  
%!        assert(prev_index<j,'Invalid prev value');
%!        assert(next_index>j,'Invalid next value');
%!  
%!        % Now we compute the ratio of interpolation:
%!        x = (j - prev_index)/(next_index - prev_index);
%!  
%!        % Now we compute the interpolated value:
%!        intdata(j) = data(prev_index)* (1.0 - x) + data(next_index) * x;
%!      else
%!        intdata(j) = data(j);
%!      end
%!    end
%!  
%!    % Now its time to compare the the value interpolated in the algorithm:
%!    % vmat(:,6)
%!    % intdata(1:size(vmat,1))
%!    difvec = abs(vmat(:,6) - intdata(1:size(vmat,1)));
%!    len = sum(difvec);
%!    assert(len==0,sprintf('Mismatch in the interpolated data values: %f',len))
%!  end

% ==> Generating continuous minute range should give continuous minute values:

%!test
%!  num = 10;
%!  dt = 1/(60*24); 
%!  for i=1:num,
%!    vbase = [random(2050),random(12),random(31),random(24)];
%!    dnum = datenum(vbase);
%!    for j=1:59,
%!      dnum += dt;
%!      vec = datevec(dnum);
%!      if vec(6)>=30,
%!        vec(5) += 1;
%!        vec(6) = 0;
%!        vec = datevec(datenum(vec));
%!      end
%!      assert(vec(6)<30,'Invalid number of seconds');
%!      min = vec(5);
%!      assert(min==j,sprintf('Mismatch in minute values: %d != %d',min,j));
%!    end
%!  end

% ==> Should be able to display progress on large dataset

%test
%  fprintf('Generating large dataset...\n')
%  n = 120000;
%  mat = gen_dataset(n,2);
%  % we should at least leave the beginning and end values of the matrix to test:
%  nh = random2(3,1000);
%
%  % now select the row indices that we want to remove:
%  % note that the resulting indices are in the range [1,n-2] so
%  % we need to add 1 to get into the range [2,n-2], and thus avoid removing the beginning and end of matrix.
%  idx = randperm(n-2,nh)+1;
%
%  % Remove the desired lines from the matrix:
%  nmat = mat;
%  nmat(idx,:) = [];
%  fprintf('Beginning long validation...\n')
%  vmat = validateDataset(nmat,true);
%  fprintf('Long validation done.\n')

% ==> Should never output an interpolation value of "1", which would means that the interpolations gap is only of 1 minute
% and thus that we should not be interpolation in the first place!

% Helper method to build inconsistent dataset:
%!function mat = inc_dataset(n,m,nh)
%!  mat = gen_dataset(n,m);
%! 
%!  % now select the row indices that we want to remove:
%!  % note that the resulting indices are in the range [1,n-2] so
%!  % we need to add 1 to get into the range [2,n-2], and thus avoid removing the beginning and end of matrix.
%!  idx = randperm(n-2,nh)+1; 
%!
%!  % Remove the desired lines from the matrix:
%!  mat(idx,:) = [];
%!endfunction

%!test
%!  n = 20000;
%!  nh = random2(10,500);
%!  mat = inc_dataset(n,2,nh);
%!  % mat = gen_dataset(n,2);
%!
%!  profile on
%!  [vmat, interp] = validateDataset(mat);
%!  profile off
%!  len = sum(interp==1);
%!  assert(len==0,sprintf('Should not find any interpolation with 1 minute length. Found len=%f',len))
%!  profshow(profile('info'))


