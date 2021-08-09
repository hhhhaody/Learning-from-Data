%% 3.3
% using the kmean function defined by 3.2
clear, clc, close all;
%% Generate Gaussian data:
mu1 = [2 2];
mu2 = [-2,2];
mu3 = [0,-3.25];
sigma1 = [0.02 0.02];
sigma2 = [0.05 0.05];
sigma3 = [0.07 0.07];
rng=('default');
data1 = mvnrnd(mu1,sigma1,50);
rng=('default');
data2 = mvnrnd(mu2,sigma2,50);
rng=('default');
data3 = mvnrnd(mu3,sigma3,50);
DATA = [data1;data2;data3];

%%
N = 10;
MU_init = [];
K = [2, 3, 4, 5, 6, 7, 8, 9, 10];
lambda = [15, 20, 25, 30];
for j = lambda(1:length(lambda))
    WCSS_result = [];
    for i = K(1:length(K))
        [~,WCSS]= kmean(DATA,i,N,MU_init,0);
        WCSS_result = [WCSS_result,WCSS+j*i]
    end
    figure();
    plot(K,WCSS_result);
    title(sprintf('Plot of WCSS + penalty λK for λ = %d',j));
    xlabel('K');
    ylabel('WCSS+penalty λK');
end
%% K-Means implementation
% Add code below

% K: # of clusters
% N: # of initializations, when N is not 1, use given MU_init
% MU_int: can initiated to anything when N is not 1, should be proper 
%         initialized when N is 1
% plot: 1 for generating plot, 0 for skip gererating plot
function [MU_final,WCSS] = kmean(DATA,K,N,MU_init,plot)
    % initialize WCSS to be inf for comparison to find minimun WCSS    
    WCSS = inf;
    for j=1:N
        
        % if # of iteration is not set to 1, random generate MUs
        if (N ~= 1)
            for i = 1:K
                MU_init(i,:) = DATA(randperm(length(DATA),1),:);
            end
        end
        
        MU_previous = MU_init;
        MU_current = MU_init;

        % initializations
        converged = 0;
        iteration = 0;
        convergence_threshold = 0.025;

        while (converged==0)
            iteration = iteration + 1;
            fprintf('Iteration: %d\n',iteration)

            %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
            % Write code below here:
            distance=diag(DATA*DATA')*ones(1,K) + ones(length(DATA),1)*diag(MU_current*MU_current')'- 2*DATA*MU_current';
            [mindist,nearest]=min(distance,[],2);

            %% CODE - Mean Updating - Update the cluster means
            % Write code below here:
            MU_previous = MU_current;
            for i=1:K
                MU_current(i,:) = mean(DATA(nearest==i,:));
            end

            %% CODE 4 - Check for convergence 
            % Write code below here:
            if (max(max(abs(MU_previous-MU_current))) < convergence_threshold)
                converged=1;
            end

            %% CODE 5 - Plot clustering results if converged:
            % Write code below here:
            if (converged == 1)
                fprintf('\nConverged.\n')

                %% If converged, get WCSS metric
                % Add code below
                
                % if current WCSS is smaller than best WCSS, update MU
                % and cluster group
                if(sum(mindist) < WCSS)
                    lables_final = nearest;
                    MU_final = MU_current;
                end
                
                % for multiple iterations, make a list for WCSS
                if(j==1)
                    WCSS_list = sum(mindist);
                else
                    WCSS_list = [WCSS_list, sum(mindist)];
                end
                
                % update best WCSS
                WCSS = min(sum(mindist),WCSS);
            end
        end
    end  
    
    % if plot required then generate the plot
    if(plot)
        figure();
        gscatter(DATA(:,1),DATA(:,2),lables_final);
        title(sprintf('kMeans clustering for %d clusters', K));
        xlabel('X');
        ylabel('Y');
    end
    
    % if iteration is not 1, generate WCSS list
    if(N~=1)
        WCSS_list
    end
    
    % report the final MU for best WCSS
    MU_final
    WCSS
end