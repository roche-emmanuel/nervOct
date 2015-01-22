
myvar = 1;

% function obj = create_object()
% 	state = 0;

% 	function increment()
% 		state += 1;
% 	end

% 	function val = getState()
% 		val = state
% 	end

% 	obj.increment = increment;
% 	obj.getState = getState;
% end

myobj1 = test_create_obj_state();
fprintf('Object1 state value is %d\n', myobj1.getState())

myobj.increment();
fprintf('Object1 state value is %d\n', myobj1.getState())

