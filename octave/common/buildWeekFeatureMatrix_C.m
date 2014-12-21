function X =  buildWeekFeatureMatrix_C(data,cfg)
% This method takes as input the raw week data containing the number of minutes and the 6 colunms of data for each dataset
% and convert that into an acceptable feature matrix using the requested number of minutes from the config argument.

% This version will only contain the close column for each dataset.

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
nc = nd;

% Now for each line, we need the number of minutes colunm, and for each symbol pair we will take the 4 price values (open,high,low,close), thus, the total number of colunms is:
m = 1 + nc * nbars;

X = zeros(n,m);

% From the input global dataset, we need to remove the colunms, we are not interested in (eg. tick volume and volume)
% These are the 2 last colunms accumulated for each symbol dataset.
% So we prepare an input matrix with only the colunms of interest:
% Note that we do not need the number of minutes in that input matrix:

inmat = zeros(nrows,nc);

for i=1:nd,
  inmat(:,i) = data(:,5+6*(i-1));
end

% Start buy setting the num minutes colunm in the feature matrix:
% Note that we need to offset the indices at the beginning and the end:
X(:,1) = indices(nbars:nbars+n-1,1);

% Now for each minute bar, we inject the sub matrix in the feature matrix:
for i=1:nbars,
  X(:,2+nc*(i-1):1+nc*i) = inmat(nbars-i+1:nbars-i+1+n-1,:);
end

end

% ==> We need a valid input dataset for this method:
%!error <must provide a dataset as input> buildWeekFeatureMatrix_C()

% ==> We need a valid configuration for this method:
%!error <must provide a configuration> buildWeekFeatureMatrix_C(rand(5,4))

% ==> Should throw an error if the input dataset does not contains the expected number of rows:
% or doesn't start with the number of minutes colunms.
%!error <Invalid number of rows in week dataset> buildWeekFeatureMatrix_C(rand(5,4),config())
%!error <Invalid minute indices in dataset> buildWeekFeatureMatrix_C(rand(config().num_week_minutes,4),config())

% ==> We can test the building of a simple feature matrix:

% Helper method to build a dummy week dataset
%!function data = build_dataset(mins_vec,cfg)
%!  % Prepare a dataset with the given number of rows, and the given number of symbol pairs.
%!  nd = cfg.num_symbol_pairs;
%!  nrows = size(mins_vec,1);
%!  data = zeros(nrows,1+nd*6);
%!  data(:,1) = mins_vec;
%!  
%!  for i=1:nd,
%!    data(:,2+6*(i-1):1+6*i) = floor(rand(nrows,6)*20);
%!  end
%!endfunction

%!test
%!  % First we need to retrieve the config struct to modify it for faster execution:
%!  cfg = config();
%!  nmins = 10;
%!  nbars = 3;
%!  npred = 2;
%!  cfg.minute_indices = (1:nmins)';
%!  cfg.num_week_minutes = nmins;
%!  
%!  % Number of minute bars to consider for each example:
%!  cfg.num_input_bars = nbars;
%!  cfg.num_pred_bars = npred;
%!  
%!  % Number of symbol pairs:
%!  cfg.num_symbol_pairs = 1;
%!  
%!  % Now we generate a simple input dataset:
%!  d = zeros(nmins,4);
%!  for i=1:4,
%!    d(:,i) = floor(rand(nmins,1)*100);
%!  end
%!  
%!  % append some random values for the tick and volume simulation:
%!  % mat = [cfg.minute_indices d rand(nmins,2)];
%!  mat = build_dataset(cfg.minute_indices,cfg);
%!  
%!  % Prepare the expected result matrix:
%!  Xe = zeros(nmins,1+4*nbars);
%!  
%!  % Inject the number of minutes:
%!  Xe(:,1)=cfg.minute_indices;
%!  
%!  % inject the data:
%!  for i=1:cfg.num_input_bars,
%!    Xe(i:end,2+4*(i-1):1+4*i) = d(1:end-i+1,:);
%!  end
%!
%!  % Remove the top and bottom lines:
%!  Xe(1:nbars-1,:) = [];
%!  Xe(end-npred+1:end,:) = [];
%!
%!  % Now compute the actual matrix:
%!  X = buildWeekFeatureMatrix_C(mat,cfg);
%!
%!  X;
%!  len = sum(abs(X-Xe));
%!  if len>0,
%!    Xe
%!    X
%!    error('Mismatch in computed and expected matrices, len=%f',len)
%!  end

% ==> We can perform additional test by matching the content of different lines in the feature matrix 
% and checking the size of the matrix:

% Helper method to return a random value between 1 and max
%!function val = random(max)
%!  val = ceil(rand(1,1)*max);
%!endfunction

% Helper method to return a random value between min and max
%!function val = random2(min, max)
%!  val = min + floor(rand(1,1)*(max-min) + 0.5);
%!endfunction

%!test
%!  num = 10;
%!  for i=1:num,
%!    cfg = config();
%!    nmins = random2(20,120);
%!    nbars = random2(3,10);
%!    npred = random2(2,5);
%!    cfg.minute_indices = (1:nmins)';
%!    cfg.num_week_minutes = nmins;
%!  
%!    % Number of minute bars to consider for each example:
%!    cfg.num_input_bars = nbars;
%!    cfg.num_pred_bars = npred;
%!  
%!    % Number of symbol pairs:
%!    nd = random2(1,5);
%!    cfg.num_symbol_pairs = nd;
%!  
%!    % Now we generate a random dataset:
%!    mat = build_dataset(cfg.minute_indices,cfg);
%!  
%!    % compute the feature matrix:
%!    X = buildWeekFeatureMatrix_C(mat,cfg);
%!  
%!    % check the size of the matrix:
%!    assert(size(X,1)==nmins-nbars-npred+1,'Invalid size for feature matrix');
%!  end

% Helper method to build a dummy week dataset
%!function inp = build_input(data,cfg)
%!  % Prepare the input matrix for a given dataset.
%!  nd = cfg.num_symbol_pairs;
%!  nrows = size(data,1);
%!  inp = zeros(nrows,nd*4);
%!  for i=1:nd,
%!    inp(:,1+4*(i-1):4*i) = data(:,2+6*(i-1):2+6*(i-1)+3);
%!  end
%!endfunction


% Check that the input we build from a given dataset is correct:
%!test
%!  num = 10;
%!  for i=1:num,
%!    cfg = config();
%!    nmins = random2(20,120);
%!    nbars = random2(3,10);
%!    npred = random2(2,5);
%!    cfg.minute_indices = (1:nmins)';
%!    cfg.num_week_minutes = nmins;
%!  
%!    % Number of minute bars to consider for each example:
%!    cfg.num_input_bars = nbars;
%!    cfg.num_pred_bars = npred;
%!  
%!    % Number of symbol pairs:
%!    nd = random2(1,6);
%!    cfg.num_symbol_pairs = nd;
%!  
%!    % Now we generate a random dataset:
%!    mat = build_dataset(cfg.minute_indices,cfg);
%!    
%!    % Prepare the input matrix:
%!    inmat = build_input(mat,cfg);
%!    
%!    snd = random2(1,nd);
%!    len = sum(abs(inmat(:,1+4*(snd-1):4*snd) - mat(:,2+6*(snd-1):2+6*(snd-1)+3)));
%!    if len>0,
%!      error('Mismatch in computed input matrix, len=%f',len)
%!    end
%!  end

% Helper method to build a dummy feature matrix
%!function X = build_feature(data,cfg)
%!  nd = cfg.num_symbol_pairs;
%!  nrows = size(data,1);
%!  inp = build_input(data,cfg);
%!  nbars = cfg.num_input_bars;
%!  npred = cfg.num_pred_bars;
%!  nc = nd*4;
%!  X = zeros(nrows,1+nc*nbars);
%!  X(:,1) = data(:,1);
%!  for i=1:nbars,
%!    X(i:end,2+nc*(i-1):2+nc*(i-1)+nc-1) = inp(1:end-i+1,:);
%!  end
%!  X(1:nbars-1,:) = [];
%!  X(end-npred+1:end,:) = [];
%!endfunction


% Check that the predicted matrix is correct:
%!test
%!  num = 10;
%!  for i=1:num,
%!    cfg = config();
%!    nmins = random2(20,120);
%!    nbars = random2(3,10);
%!    npred = random2(2,5);
%!    cfg.minute_indices = (1:nmins)';
%!    cfg.num_week_minutes = nmins;
%!  
%!    % Number of minute bars to consider for each example:
%!    cfg.num_input_bars = nbars;
%!    cfg.num_pred_bars = npred;
%!  
%!    % Number of symbol pairs:
%!    nd = random2(1,6);
%!    cfg.num_symbol_pairs = nd;
%!  
%!    % Now we generate a random dataset:
%!    mat = build_dataset(cfg.minute_indices,cfg);
%!    %mat
%!
%!    % compute the feature matrix:
%!    X = buildWeekFeatureMatrix_C(mat,cfg);
%!    %X
%!
%!    % Prepare the expected feture matrix:
%!    Xe = build_feature(mat,cfg);
%!    %Xe
%!    
%!    len = sum(abs(Xe - X));
%!    if len>0,
%!      X
%!      Xe
%!      error('Mismatch in computed input matrix, len=%f',len)
%!    end
%!  end

% ==> Now test performances with a real size dataset:

%test
%  cfg = config();
%  nd = cfg.num_symbol_pairs;
%  nrows = cfg.num_week_minutes
%  mat = build_dataset(cfg.minute_indices,cfg);
%  tic();
%  profile on;
%  
%  % compute the feature matrix:
%  X = buildWeekFeatureMatrix_C(mat,cfg);
%  profile off;
%  
%  toc()
%  profshow(profile('info'))
%  size(X)