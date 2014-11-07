function data = loadData(dpath,fname,vname)
% load a data matrix from a file with name fname'.txt/.mat'
% it the .mat file doesn't exist yet it is created to speed up future loadings.
% then the variable name vname is used in that case.

assert(exist('dpath', 'var') == 1, 'must provide data path')
assert(exist('fname', 'var') == 1, 'must provide file name')
assert(exist('vname', 'var') == 1, 'must provide variable name')

% append final '/' if needed
datapath = dpath;
% fprintf('Checking datapath %s\n', datapath)
res = strchr(datapath, '/');
% res = regexp(datapath, '/$');
% size(res,2)
% if size(rec,2)==0,
if res(end)~=length(datapath),
	datapath = [datapath '/'];
	% fprintf('Fixed datapath %s\n', datapath)
end

% check if the file datapath+fname+.mat exists:
% tic()
matfile = [datapath fname '.mat'];

[info, err, msg] = stat(matfile);

if err==-1,
	fprintf('Creating binary file for %s...\n',fname)
	txtfile = [datapath fname '.txt'];

	% Check that the text file exists:
	[info, err, msg] = stat(txtfile);
	assert(err==0,['Cannot find file ' txtfile])
	
	% Load the data:
	eval([vname '=load("' txtfile '");'])

	% Save the data:
	eval(['save -binary "' matfile '" ' vname ';'])
end

% Now we can load the mat file as expected:
loaded = load(matfile,vname);

data = loaded.(vname);
% toc()

end

%!error <must provide data path> loadData()
%!error <must provide file name> loadData('data')
%!error <must provide variable name> loadData('data','file')
%!test
%!  mkdir test;
%!  num = 10;
%!	rsize = 5;
%!	csize = 6;
%!  for i=1:num,
%!  	m = rand(rsize,csize);
%!  	% Remove any previous file:
%!		unlink 'test/my_data.txt';
%!		unlink 'test/my_data.mat';
%!  	% Write this as a text file:
%!    csvwrite('test/my_data.txt',m);
%!		m2 = loadData('test/','my_data','m');
%!		[info, err, msg] = stat('test/my_data.mat');
%!		assert(err==0,'Could not create mat file.')
%!		% Check that the loaded matrix contains the same data:
%!		assert(size(m)==size(m2));
%!		for r=1:rsize,
%!    	for c=1:csize,
%!				if abs(m(r,c)-m2(r,c))>=1e-10,
%!					fprintf('%f != %f at row=%d, col=%d\n',m(r,c),m2(r,c),r,c);
%!				end
%!				assert(abs(m(r,c)-m2(r,c))<1e-10,'Saved values do not match');
%!			end
%!		end
%!  end
%!  unlink 'test/my_data.txt';
%!  unlink 'test/my_data.mat';
%!  rmdir 'test';
