
cont.obj.name = 'my_object';
cont.obj.index = 0;

function set_struct_var(box)
	if (!isglobal('my_counter'))
		global my_counter = 0;
	end

	global my_counter;
	my_counter++;

	fprintf('Counter value is %d\n', my_counter)
	box.obj.index = my_counter;
end

set_struct_var(cont);
fprintf('Object index value is %d\n', cont.obj.index)

set_struct_var(cont);
fprintf('Object index value is %d\n', cont.obj.index)
