
% The prototype to call the function should be:
% [results] = trade_strategy(cmd,id,desc)

% ==> Should throw an error if no argument is provided:
%!error <trade_strategy: Invalid number of arguments: 0> trade_strategy()

% ==> Should throw an error if command is not a string
%!error <trade_strategy: command is not a string> trade_strategy(1,0,3)

% ==> Should throw an error if command is not known:

%!function invalid_command()
%!  desc.dummy = 10;
%!	trade_strategy('dummy',uint32(1),desc)
%!endfunction

%!error <trade_strategy: unknown command name: dummy> invalid_command()

% ==> Should throw an error if create is called without valid traits:
%!error <trade_strategy: desc should be a structure type> trade_strategy('create',3)

% ==> Should throw an error if creation traits do not contain target_symbol:

%!function invalid_creation()
%!  desc.dummy = 10;
%!	trade_strategy('create',desc)
%!endfunction

%!error <trade_stategy: target_symbol value is not defined.> invalid_creation()

% ==> It should create a valid strategy when the traits are OK:
%!test
%!	desc.target_symbol = 6;
%!	sid = trade_strategy('create',desc)
%!	assert(sid>0,'Invalid valud for Strategy id: %d',sid)
%!	% once a strategy is create it should be possible to destroy it:
%!	trade_strategy('destroy',sid)

% ==> It should be possible to evaluate a strategy:
%!test
%!	cdesc.target_symbol = 6;
%!	sid = trade_strategy('create',cdesc)
%!	
%!	% Prepare the evaluation traits:
%!	nf = 1 + 4*6*3;
%!	nsamples = 100;
%!	evdesc.inputs = rand(nf,nsamples);
%!	
%!	% Perform evaluation:
%!	trade_strategy('evaluate',sid,evdesc);
%!
%!	% once a strategy is create it should be possible to destroy it:
%!	trade_strategy('destroy',sid)
