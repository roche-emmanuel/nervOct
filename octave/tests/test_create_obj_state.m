function obj = test_create_obj_state()
	state = 0;

	function increment()
		state += 1;
	end

	function val = getState()
		val = state
	end

	obj.increment = @increment;
	obj.getState = @getState;
end
