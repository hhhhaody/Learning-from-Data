% EC 503 - HW 3 - Fall 2020
% K-Means starter code

clear, clc, close all;

%% Generate Gaussian data:
% Add code below:
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


figure();
plot(data1(:,1),data1(:, 2),'r+');
hold on;
plot(data2(:,1),data2(:,2),'g+');
plot(data3(:,1),data3(:,2),'b+');
title('3 Gaussian Clusters');
xlabel('X');
ylabel('Y');
legend('1','2','3','location','EastOutside');
hold off;
%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here
NBAdata = readmatrix('NBA_stats_2018_2019.xlsx');
NBA=[NBAdata(:,5),NBAdata(:,7)];

% Problem 3.2(f): Generate Concentric Rings Dataset using
% sample_circle.m provided to you in the HW 3 folder on Blackboard.
[circledata,circlelabel] = sample_circle(3,[500,500,500]);

%% 3.2(a)
disp('3.2a');
K = 3;
N = 1;
MU_init = [3,3;-4,-1;2,-4];
kmean(DATA,K,N,MU_init,1);

%% 3.2(b)
disp('3.2b');
K = 3;
N = 1;
MU_init = [-0.14,2.61;3.15,-0.84;-3.28,-1.58];
kmean(DATA,K,N,MU_init,1);

%% 3.2(c)
disp('3.2c');
K = 3;
N = 10;
MU_init = [];
kmean(DATA,K,N,MU_init,1);

%% 3.2(d)
disp('3.2d');
N = 10;
MU_init = [];
K = [2, 3, 4, 5, 6, 7, 8, 9, 10];
WCSS_result = [];
for i = K(1:length(K))
    [~,WCSS]= kmean(DATA,i,N,MU_init,0);
    WCSS_result = [WCSS_result,WCSS];
end
figure();
plot(K,WCSS_result);
title('Plot of WCSS for different K');
xlabel('K');
ylabel('WCSS');

%% 3.2(e)
disp('3.2e');

figure();
scatter(NBA(:,1),NBA(:,2));
xlabel('MPG');
ylabel('PPG');
title('Scatterplot of NBA Data');

K = 10;
N = 10;
MU_init = [];
kmean(NBA,K,N,MU_init,1);
xlabel('MPG');
ylabel('PPG');

%% 3.2(f)
figure();
gscatter(circledata(:,1),circledata(:,2),circlelabel);
xlabel('X');
ylabel('Y');
title('Scatterplot of Concentric Data');

K = 3;
N = 10;
MU_init = [];
kmean(circledata,K,N,MU_init,1);
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
