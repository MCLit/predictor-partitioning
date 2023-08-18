function [group_0, label_0, group_k, label_k] = prepare_g0(X, PD)
% input
% portion - % dataset that will use as the remain inside X
% X - the dataset
% output: group_0 - a portion of the data that can be used to obtain q
% group_0s,
% remaining_X = the portion of hte data that will be further subdivided and
% produce 2^k groups
% thread: https://uk.mathworks.com/matlabcentral/answers/388385-how-can-i-do-a-80-20-split-on-datasets-to-obtain-training-and-test-datasets

s = size(X,2);
labels =compose('x%g',[1:s]); % make labels of the whole dataset

cv = cvpartition(size(X,2),'HoldOut',PD);

group_k = X(:,cv.training);
label_k = labels(:,cv.training);

group_0 = X(:,cv.test);
label_0 = labels(:,cv.test);

end