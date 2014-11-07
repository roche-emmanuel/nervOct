function X =  buildFeatureMatrix(weeks_dataset,cfg)
% This method will build the feature matrix for all the available weeks
% And then save the result in a new file.
% note that it will also compute the label matrix at the same time and save it it the same file.

nweeks = size(weeks_dataset,2);
dpath = cfg.datapath;
dname = cfg.week_feature_pattern;
fpattern = [dpath '/' dname '.mat'];

h = waitbar(0.0);

fprintf('Generating feature matrices for all weeks...\n')

for i=1:nweeks,
  if isempty(weeks_dataset{1,i}),
    continue;
  end

  % fprintf('Generating feature matrix for week %d...\n',i)
  waitbar(i/nweeks,h,sprintf('Processing week %d... (%02d%%)\n',i,floor(0.5 + 100*i/nweeks)));  

  % prepare the name for this feature file:
  fname = sprintf(fpattern,i);

  % First check if the file exist (and thus should be deleted):
  [info, err, msg] = stat(fname);

  if err==0,
    % delete the previous file:
    fprintf('Deleting previous feature file %s...\n',fname);
    unlink(fname);
  end

  data = weeks_dataset{1,i};
  week_features = buildWeekFeatureMatrix(data,cfg); 
  week_labels = buildWeekLabelMatrix(data,cfg); 

  assert(size(week_features,1)==size(week_labels,1),'Mismatch in feature and label matrices.')

  save('-binary',fname,'week_features','week_labels');

  % eval(sprintf('week_%d_features = X;',i));
  % Now save the result before continuing:
  % vname = sprintf('week_%d_features',i);
  % save('-binary','-append',fname,vname);
  % clear(vname);
end

close(h);

fprintf('Week feature generation done.\n')

end
