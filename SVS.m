

function mdl = SVS(X, y, labels) % Simple Variable Selection
% Inputs: X - observations by variables matrix of candidate predictors
%         y - the outcome variable vector
%         labels - labels of the candidate predictors
% starts with regression composed of constant term, finds linear terms that
% belong to the model, based on term's BIC value (now 0), does not produce
% any output text and uses variable names provided by main code
    mdl = stepwiselm(X, y, 'constant', 'upper', 'linear', 'criterion', 'aic', 'penter', -3, 'Verbose', 0, 'VarNames', [labels, 'y']); % linear regression
%     IBIC = mdl.ModelCriterion.BIC;
    % for backward selection see this:
    % https://stackoverflow.com/questions/11040006/backward-elimination-technique-in-matlab
end