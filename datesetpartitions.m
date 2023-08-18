function [dataGroups, labelGroups] = datesetpartitions(X, label, q)
% In: matrix to divide, labels of each column, number of groups to generate
% thread https://uk.mathworks.com/matlabcentral/answers/484662-randomly-grouping-cells-in-cell-array
names = label;    % {'a' 'b' 'c' ... 'v'} 
number_of_people = numel(names); 
number_of_groups = q; 
% Create random order of names
randOrder = randperm(numel(names)); 
namesPerm = names(randOrder); 
out = X(:,randOrder);
% Determine number of people per group
groupNum = floor(number_of_people / number_of_groups) + [ones(1,rem(number_of_people,number_of_groups)), zeros(1,number_of_groups - rem(number_of_people,number_of_groups))];
% Create groups
groupIdx = [0,cumsum(groupNum)]; 
labelGroups = arrayfun(@(i){namesPerm(groupIdx(i)+1:groupIdx(i+1))},1:numel(groupNum));
dataGroups = arrayfun(@(i){out(:,groupIdx(i)+1:groupIdx(i+1))},1:numel(groupNum));
dataGroups = dataGroups';
end % randomly allocates the data into k groups