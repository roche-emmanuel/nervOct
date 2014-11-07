% method used to generate an enumeration:
function [varargout] = enum (first_index = 1) 
  for k = 1:nargout 
    varargout {k} = k + first_index - 1; 
  endfor 
end
