function require_functions(function_names)
%REQUIRE_FUNCTIONS Ensure required functions exist on the MATLAB path.
%
% require_functions({"ea_getptopts", "ea_map_coords"})
%

if ischar(function_names) || isstring(function_names)
    names = string(function_names);
else
    names = string(function_names);
end

for i = 1:numel(names)
    fn = char(names(i));
    if exist(fn, 'file') ~= 2
        error(['Required function "%s" not found on MATLAB path. ', ...
               'Please add Lead-DBS to the MATLAB path before running.'], fn);
    end
end

end
