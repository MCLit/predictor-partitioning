% This script conducts forwards and backwards stepwise analysis
% according to subset sampling alrogithm described in:
% https://stackoverflow.com/questions/11040006/backward-elimination-technique-in-matlab
% Process: 
% - initial comparison is defined as a null model (only composed of
% constant)
% - on the first iteration, the set of X features is split into two equally sized groups
% (features are randomly assigned to each), one group is used to produce q group_0s
% the other group is then split into k groups
% - Nelson's script is used to obtain all possible combinations of group_0 with group_k
% - we apply stepwise regression to each combination of groups_0 and groups_k
% - we find the minimum model evidence to find the winning model for each
% of q entertained models (minimum AIC)
% - if the new minimum AIC is smaller than the AIC from the
% last iteration, then the q winning models are saved in variable new_group_0_mdls
% - we then ensure that this is not a local solution by randomly partitioning the data 
% and applying the SVS algorithm again
% - this is done 10-20 times
% - if at any point an improved AIC value is found then it takes over the
% new_group_0_mdls variable and we move to the next iteration
% - on the following iteration, the elements from each new_group_0_mdls are used
% as q group_0 and the SVS algorithm is applied again
% - variables that are not included in any of the new_group_0_mdls are
% partitioned into k groups

clc
clear

load('sample_data.mat')
% variable setup:
[coef,score,latent,~,explained,~] = pca(deg_f,'Centered',true);

comps = 1; % number of components
for i = 1:length(explained) % this loop finds the cumulative variance
    s = sum(explained([1:i]));
    if  s <= 100
        comps = comps +1 ;
    end
end

X = score(:,[1:comps]);
y = WM; % working memory component

K = 4; % number of groups
q = 15; % number of entertained models
s = 2^K; % total number of grouping schemes
groupind = grouppartitions(K); % Nelson's script to produce a logical structure with all possible grouping schemes 

% initial model for iteration 1 to produce a comparison
Group_0 = ones(249, 1);
mdl_0 = stepwiselm(Group_0, y, 'Verbose', 0); % the initial AIC
AIC_0 = mdl_0.ModelCriterion.AIC; % this is a placeholder for the winning model

% loop preparations
ISET = cell(q,1);
flag = 0; % flag to stop the search once improved model was found

% script output will be either in variable mdl_0 if we do not find a good solution
% or in variable winner, if we do find a solution that is robust
% the best AIC is always stored in variable AIC_0

for j = 1:1000 % number of iterations depends on our computing power
    
%%% DATA PREPARATION - partition the data into q group_0s and k group_ks

    if j == 1 % on the first iteration, Group_0 is produced randomly
        % 1: we divide the dataset into 2 equal parts (50%/50% of whole dataset)
        [group_0, label_0, group_k, label_k] = prepare_g0(X, 0.50);
        % 2: we divide group_0 into q candidate models
        [group_q, label_q] = datesetpartitions(group_0, label_0, q);
        % 3: we divide the remaining 50% of the data into K subsets
        [group_k, label_k] = datesetpartitions(group_k, label_k, K);
        group_k = group_k'; group_q = group_q'; % transposition for consistent dimensions

    else % on subsequent iterations, q group_0s are the variables from the models from the last iteration of the SVS algorithm 
        % that identified an imporved AIC (best = lowest AIC value) - see line 142 onwards for decision whether group_0 changes
        for d = 1:q % q group_0s
            tmp_X = table2array(new_group_0_mdls{d,1}.Variables); % previous iteration's candidate predictors for each q winning model
            in_model = new_group_0_mdls{d,1}.Formula.InModel; % list which features were included in the model 
            group_q{1,d} = tmp_X(:,[in_model == 1]); % the new group_q select features that were selected as predictors
            label_q{1,d} = new_group_0_mdls{d,1}.VariableNames([in_model == 1]); % their labels
            % Note: Component N can be present in more than one group 0
        end 
        
        % k groups contain variables not in any of the group_0s
        tmp_labels_q = vertcat(label_q{:}); % these will be to indces we are dropping
        tmp_labels_q = unique(tmp_labels_q);
        tmp_labels_q = extractAfter(tmp_labels_q, "x");
        tmp_labels_q = str2double(tmp_labels_q);
%%% Note: it can happen that Component N is considered by q models 1
        % and 3 because it is part of their group_0, but this component
        % will not be available to model 2 and 4, as it is not part of any
        % group_k
        
        group_k = X;
        group_k(:,[tmp_labels_q]) = []; % only keep those elements that are not part of group_0 
        
        sz = size(X,2);
        label_k =compose('x%g',[1:sz]);
        label_k(:,[tmp_labels_q]) = [];
        [group_k, label_k] = datesetpartitions(group_k, label_k, K);
        group_k = group_k'; label_q = cellfun(@transpose,label_q,'un',0);  % transposition to fit dimentions
    end
    
%%% conduct stepwise analysis to find q by s sets

    for i = 1:q
        for s = 1:size(groupind,1) % for each combination of group_0 and group_k subset combination
            % select current subset selection
            
            current_groupind = groupind(s,:);
            group_s = group_k(:,[current_groupind==1]);
            label_s = label_k(:,[current_groupind==1]);
            % combine group_0 and the subset

            % taking data out of cells, I hate cell indexing
            g_q = group_q{1,i};
            group_s = group_s{1,1};
            X_s = [g_q group_s];
            labels_s = label_s{1,1};
            l = label_q{1,i};
            labels = [l, labels_s];
            
            ISET{i,s} = SVS(X_s, y, labels); % Simple Variable Selection (SVS) - stepwise linear regression
        end
    end % we now have q group_0 by s subset combinations 

%%% find the winning model
               
        for d = 1:i % place AIC values of each model into a double
            for e = 1:s
                candidates(d,e) = ISET{d,e}.ModelCriterion.AIC; 
            end
        end
        
       [candidate_values, candidate_col] = min(candidates,[],2); % find minimum AIC value for each q models 
       [candidate_value, candidate_row] = min(candidate_values); % find where we have minimum AIC among these combinations
       
        
%%% ensure that this is not a local solution
        
        if  candidate_value > AIC_0 % if the found solution is greater than previous
            mdl_0 = mdl_0; % abandon this run and move to j+1 without changing new_group_0_mdls
            
        else % if previous AIC value was greater and we have now improved the model
            mdl_0 = ISET{candidate_row,candidate_col(candidate_row)}; % update mdl_0 with the new model 
            AIC_0 = candidate_value; 
            
            for t = 1:q % next iteration will use this iteration's q winning models to make informed group_0
                new_group_0_mdls(t,1) = ISET(t,candidate_col(t));
            end
            % and check that this is not a local solution
            for v = 1:10 % produce 10 more iterations, as advised, lines 164 to 207 basically repeat the above steps and all variables have 'c' suffix to note that they correspond to the above variables
                    [group_0c, label_0c, group_kc, label_kc] = prepare_g0(X, 0.50);
                    [group_qc, label_qc] = datesetpartitions(group_0c, label_0c, q);
                    [group_kc, label_kc] = datesetpartitions(group_kc, label_kc, K);
                    group_kc = group_kc'; group_qc = group_qc';

                for h = 1:q % number of entertained alternative models
                    for g = 1:size(groupind,1) % for each group allocation

                        % select current subset selection
                        current_groupindc = groupind(g,:);
                        group_sc = group_kc(:,[current_groupindc==1]);
                        label_sc = label_kc(:,[current_groupindc==1]);
                        % combine group_0 and the subset
                        g_qc = group_qc{1,h};% taking data out of cells
                        group_sc = group_sc{1,1};
                        X_sc = [g_qc group_sc];
                        labelsc = [label_qc{1,h} label_sc{1,1}];
                        vmdl{h,g} = SVS(X_sc, y, labelsc);
                    end
                end % we now have q group_0 models and s subsets

                for d = 1:h % we know how many candidate group_0 we have entertained in the end
                    for e = 1:g % and we know how many subsets there were in the end
                        vAIC_k(d,e) = vmdl{d,e}.ModelCriterion.AIC; % place AIC values of each model into double
                    end
                end
            end


                [minv_value_col, minv_index_min_col] = min(vAIC_k,[],2);  
                [minv_value, minv_index_min_row] = min(minv_value_col); % find where we have minimum AIC among these combinations

                if minv_value < AIC_0
                    mdl_0 = vmdl{minv_index_min_row, minv_index_min_col(minv_index_min_row)};

                    for t = 1:q
                            new_group_0_mdls(t,1) = vmdl(t,minv_index_min_col(t,1));
                    end

                    % and we continue on with the main analysis pipeline
                else % if this is not a local solution
                    winner = ISET(candidate_row,candidate_col(candidate_row,1));
                    winner = winner{1,1};
                    % and we end the analysis
                    disp('ISET found successfully')
                    flag=1;
                    break
                end
            % then and group0 must change
            % else and keep group 0 as was
            
            if (flag == 1) % set has been found successfully and we do not have to continue our search
                break 
            end
        end
end

disp('Analysis')
disp('entertained group_0 models')
disp(q)
disp('number of groups')
disp(K)
disp('number of subset combinations with each group_0')
disp(s)
disp('winner was found:')
disp(flag)
disp('number of iterations')
disp(j)
disp('winning model')

if flag == 1
    mdl_0
else
    AIC_0
end

% if flag = 1 then the winning model is inside winner
% else the best model so far is inside mdl_0 and the AIC value that could
% not be surpassed is in AIC_0

