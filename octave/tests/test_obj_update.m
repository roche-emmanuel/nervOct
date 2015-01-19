

function obj = increment_object(obj,step)
	obj.index = obj.index + step;
end

function obj = create_object()
	obj.id = 'my_object';
	obj.index = 0;
	obj.increment = @(step)(increment_object(obj,step))
end

myobj = create_object();
fprintf('Object index value is %d\n', myobj.index)

myobj = myobj.increment(1);
fprintf('Object index value is %d\n', myobj.index)

myobj = myobj.increment(1);
fprintf('Object index value is %d\n', myobj.index)
