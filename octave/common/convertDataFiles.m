function convertDataFiles(dpath, fpattern)
% Will write a mat file for each file found in the given directory matching the given pattern

% more off;
% tic()
pname = [pwd() '/' dpath];
filelist = readdir (pname)
for ii = 1:numel(filelist)
  % skip special files . and ..
  if (regexp (filelist{ii}, '^\\.\\.?$'))
    % fprintf('Discarding file %s\n', filelist{ii})
    continue;
  end

  % chekc if the file is a text file:
  [dirname, fname, ext, vername] = fileparts (filelist{ii});
  % fprintf('Found extension %s\n',ext);
  if strcmp(ext,'.txt')==0
  	continue;
  end

  % check if the corresponding mat file already exists:
  [info, err, msg] = stat([dpath '/' fname '.mat']);
  if err==0
    fprintf('Discarding file %s (=> .mat file exists)\n', filelist{ii})
    continue; % We do not want to override an existing mat file.
  end

  % load the file
  if (regexp (filelist{ii}, fpattern))
  	fprintf('Processing file %s...\n',filelist{ii})
  	loadData([dpath '/'],fname,'dataset');
  end
end

fprintf('Done converting files.\n')

% toc()
% more on;

end
