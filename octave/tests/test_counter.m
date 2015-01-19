function test_counter()
	if (!isglobal('my_counter'))
		global my_counter = 0;
	end

	global my_counter;
	my_counter++;
	fprintf('Counter value is %d\n', my_counter)
end
