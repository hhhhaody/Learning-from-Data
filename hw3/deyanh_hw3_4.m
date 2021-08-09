% EC 503 - HW 3 - Fall 2020
% DP-Means starter code

clear, clc, close all,

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

%% Generate NBA data:
% Add code below:
NBAdata = readmatrix('NBA_stats_2018_2019.xlsx');
NBA=[NBAdata(:,5),NBAdata(:,7)];
% HINT: readmatrix might be useful here

%% 3.4(b)
LAMBDA = [0.15, 0.4, 3, 20];
for i = LAMBDA(1:length(LAMBDA))
    DPmeans(DATA, i, 1);
end
%% 3.4(c)
LAMBDA = [44, 100, 450];
for i = LAMBDA(1:length(LAMBDA))
    DPmeans(NBA, i, 1);
    xlabel('MPG');
    ylabel('PPG');
end
%% DP Means method:
function [K, MU] = DPmeans(DATA,LAMBDA,plot)
% Parameter Initializations
num_points = length(DATA);
convergence_threshold = 1;
total_indices = [1:num_points];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DP Means - Initializations for algorithm %%%
% cluster count
K = 1;

% sets of points that make up clusters
L = {};
L = [L [1:num_points]];

    % Class indicators/labels
    Z = ones(1,num_points);

    % means
    MU = [];
    MU = [MU; mean(DATA,1)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initializations for algorithm:
    converged = 0;
    t = 0;
    while (converged == 0)
        t = t + 1;
        K_previous = K;
        MU_previous = MU;
        fprintf('Current iteration: %d...\n',t)

        %% Per Data Point:
        for i = 1:num_points

            %% CODE 1 - Calculate distance from current point to all currently existing clusters
            % Write code below here:
            distance = diag(DATA(i,:)*DATA(i,:)')+ diag(MU*MU')' - 2*DATA(i,:)*MU';
            %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
            % Write code below here:
            [mindist,nearest]=min(distance,[],2);
            if (min(mindist)>LAMBDA)
                K= K+1;
                Z(i)=K;
                MU=[MU; DATA(i,:)];
            else
                Z(i)= nearest;
            end

        end

        %% CODE 3 - Form new sets of points (clusters)
        % Write code below here:
        distance = diag(DATA*DATA')+ diag(MU*MU')' - 2*DATA*MU';
        [mindist,nearest]=min(distance,[],2);

        %% CODE 4 - Recompute means per cluster
        % Write code below here:
        for j=1:K
            if (~isempty(DATA(nearest == i,:)))
                MU(j,:) = mean(DATA(nearest==j,:));
            end
        end
        %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
        % Write code below here:
        if((K_previous == K) && (max(max(abs(MU_previous-MU))) < convergence_threshold))
            converged = 1;
        end
        %% CODE 6 - Plot final clusters after convergence 
        % Write code below here:

        if (converged)
            %%%%
            figure();
            gscatter(DATA(:,1),DATA(:,2),nearest);
            title(sprintf('DPMeans clustering for %i λ', LAMBDA));
            xlabel('X');
            ylabel('Y');
        end    
    end
end

