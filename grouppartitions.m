
function groupind = grouppartitions(K)

% This function creates logical vectors indexing all possible combinations
% among K elements. This is used to create all possible combinations of
% groupings of regressors in the algorithm developed in [1].
%
% Let's assume we have N regressors in a regression model and we partition
% those regressors into K+1 groups Group0, Group1,..., GroupK, so that each
% Group is a non-overlapping subset of the N regressors. We now are
% interested in exploring all models formed by combining Group0 with all
% possible combinations of the other K Groups. There will be 2^K-1 model
% possibilities in total. We can associate each combination of the K Groups
% with a logical vector (i.e. a vector of 1s and 0s), of length K, with 1
% in the position associated with the Groups being considered in a Model
% and 0 in the position of the Groups left out. For example, if we have K =
% 4 (i.e. K Groups in adddition to Group0), then the logical
%
% [0 1 1 0]
%
% Denotes a Model where Group2 and Group3 are included and Group1 and
% Group4 are not.
%
% This function generates a logical matrix, where each row represent one of
% the mossible models to be explored, i.e. each row is one instance of the
% logical vectors exemplified above. Therefore, one can iterate accross
% rows to explore all the possible Models (each represented by one logical
% vector).
%
%   Inputs:
%       K-  Number of groups in which the N regressors are partitioned, so
%           that the total number of groups including Group0 is K+1
%
%   Outputs:
%       groupind-   Matrix of logical vectors indexing the combination of
%                   regressors used in each Model possibility. Size =
%                   [Nmodels X K]
%
%   Reference:
%   [1] Airong Cai, Ruey S. Tsay & Rong Chen (2009) Variable Selection in
%       Linear Regression With Many Predictors, Journal of Computational
%       and Graphical Statistics, 18:3, 573-591, DOI:
%       10.1198/jcgs.2009.06164
%
%   Author:
%       Nelson J trujillo-Barreto
%   Date:
%       28/05/2021

Ngroups = 2^K-1;
binchar = cellstr(dec2bin([1:Ngroups]',K));
binstr = strcat(binchar{:});
binstr = arrayfun(@(x) insertBefore(x, x, ' '), binstr, 'UniformOutput',false);
binstr = strtrim(strcat(binstr{:}));

groupind = logical(str2num(strtrim(strcat(binstr))));
groupind = reshape(groupind,K,Ngroups)';

end