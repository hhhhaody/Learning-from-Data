% EC 503 Learning from Data
% Fall semester, 2020
% Homework 2
% by (Deyan Hao)
%
% Nearest Neighbor Classifier
%
% Problem 2.5 a, b, c, d


clc, clear

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

%% a) Plotting
% include a scatter plot
% MATLAB function: gscatter()

figure()
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain)
% label axis and include title
xlabel('X axis')
ylabel('Y axis')
title('Scatter Plot Of Training Data')



%% b)Plotting Probabilities on a 2D map
K = 10;
% specify grid
[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);

% compute distance between every test points and train points
distance=diag(Xtest*Xtest')*ones(1,Ntrain) + ones(Ntest,1)*diag(Xtrain*Xtrain')'- 2*Xtest*Xtrain';

% select K nearest points and find the lable with most occurance
[~,nearest]=sort(distance,2);
nearest = nearest(:,1:K);
y_pred = ytrain(nearest);


% compute probabilities of being in class 2 for each point on grid
probabilities = sum(y_pred==2,2)/K;

% Figure for class 2
figure
class2ProbonGrid = reshape(probabilities,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('X axis')
ylabel('Y axis')
title('Probability of Being Classified as Class 2')


% repeat steps above for class 3 below
probabilities3 = sum(y_pred==3,2)/K;

% Figure for class 3
figure
class3ProbonGrid = reshape(probabilities3,size(Xgrid));
contourf(Xgrid,Ygrid,class3ProbonGrid);
colorbar;

% remember to include title and labels!
xlabel('X axis')
ylabel('Y axis')
title('Probability of Being Classified as Class 3')

%% c) Class label predictions
K = 1 ; % K = 1 case

% compute predictions 
ypred = ytrain(nearest(:,1:K));
figure
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
xlabel('X axis')
ylabel('Y axis')
title('1NN Prediction')

% repeat steps above for the K=5 case. Include code for this below.
K = 5;

% compute predictions
ypred = mode(ytrain(nearest(:,1:K)),2);

% draw picture
figure
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
xlabel('X axis')
ylabel('Y axis')
title('5NN Prediction')

%% d) LOOCV CCR computations

% compute distance for every point in training set from others and sort it
distance = diag(Xtrain*Xtrain')*ones(1,Ntrain) + ones(Ntrain,1)*diag(Xtrain*Xtrain')'- 2*Xtrain*Xtrain';
[~,nearest]=sort(distance,2);
nearest(:,1)=[]; % exclude itself from the closest points

for k = 1:2:11
    % determine leave-one-out predictions for k
    ypred = mode(ytrain(nearest(:,1:k)),2);


    % compute confusion matrix
    conf_mat = confusionmat(ytrain, ypred);
    % from confusion matrix, compute CCR
    CCR = sum(diag(conf_mat))/Ntrain;
    
    % below is logic for collecting CCRs into one vector
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end
end

% plot CCR values for k = 1,3,5,7,9,11
% label x/y axes and include title
plot([1,3,5,7,9,11],CCR_values)
xlabel('K')
ylabel('LOOCV CCR')
title('LOOCV CCR For Different K')

[~,index]=max(CCR_values); % find the index of k which maximize CCR
choosek=min(2*index-1)