
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
%!error <trade_strategy: desc should be a structure type> trade_strategy('create',uint32(0),3)

% ==> Should return a valid id when we request the creation of a strategy:
%test
%	id = trade_strategy('create')