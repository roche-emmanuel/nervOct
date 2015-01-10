% This script is used t demonstrate the global setup of the data
% needed to perform investigations.

% Initialization
clear; close all; clc
more off;

% First we add the common location path:
pname = pwd();

arch=computer();
if strcmp(arch,'x86_64-w64-mingw32')==1
fprintf('Testing on x64 architecture.\n')
addpath([pname '/../bin/x64']); %add the binary folder.
else
fprintf('Testing on x86 architecture.\n')
addpath([pname '/../bin/x86']); %add the binary folder.
end

% addpath([pname '/common']);
% addpath([pname '/neural']);
addpath([pname '/ufldl/minFunc']);
addpath([pname '/ufldl/ex5']);
addpath([pname '/ufldl/ex4']);
addpath([pname '/ufldl/ex2']);

stlExercise

more on;
