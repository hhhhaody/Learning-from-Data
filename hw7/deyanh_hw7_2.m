%% 7.2 Data Process
clear;clc;
load('iris.mat')
rng('default')
n_train = length(X_data_train);
n_test = length(X_data_test);
d = 2;
m = 3;
X_train = [X_data_train(:,2),X_data_train(:,4)];
X_test = [X_data_test(:,2),X_data_test(:,4)];
x_train=[X_train(1:70,:);[X_train(1:35,:);X_train(71:105,:)];X_train(36:105,:)];
x_test=[X_test(1:30,:);[X_test(1:15,:);X_test(31:45,:)];X_test(16:45,:)];
y_train = [ones(35,1);-1*ones(35,1)];
y_test = [ones(15,1);-1*ones(15,1)];
ntrain = length(y_train);
ntest = length(y_test);
    
%% Algorithm
theta = zeros(d+1,3);
tmax = 2*10^5;
C = 1.2;
X_train_ext=[x_train,ones(length(x_train),1)];
X_test_ext=[x_test,ones(length(x_test),1)];
norm_cost = zeros(3,200);
ccr_train = zeros(3,200);
ccr_test = zeros(3,200);
ypred_train = zeros(ntrain,3);
ypred_test = zeros(ntest,3);

for i = 1:3
    itrain=[1:ntrain:length(x_train)];
    itest=[1:ntest:length(x_test)];
    x_train_ext = X_train_ext(itrain(i):(i*70),:);
    x_test_ext = X_test_ext(itest(i):(i*30),:);
    for t = 1:tmax
        j = randi([1,ntrain]);
        v = [theta(1:d,i);0];
        if (y_train(j)*theta(:,i)'*x_train_ext(j,:)') < 1
            v = v - ntrain*C*y_train(j)*x_train_ext(j,:)';
        end
        theta(:,i) = theta(:,i) - (0.5/t)*v;
        
        if mod(t,1000) == 0
            f_0 = sqrt(theta(1,i)^2+theta(2,i)^2)/2;
            f_j = 0;
            ypred = zeros(ntrain,1);
            for n = 1:ntrain
                f_j = f_j + C*max(0,1-y_train(n)*theta(:,i)'*x_train_ext(n,:)');
            end
            
            train_pred=sign(theta(:,i)'*x_train_ext');
            ccr_train(i,t/1000) = sum(train_pred'==y_train)/ntrain;
            ypred_train(:,i) = train_pred';
            test_pred=sign(theta(:,i)'*x_test_ext');
            ccr_test(i,t/1000) = sum(test_pred'==y_test)/ntest;
            ypred_test(:,i) = test_pred';
            norm_cost(i,t/1000) = (1/ntrain)*(f_0+f_j);
        end     
    end
end

%% a
class = {'1 and 2','1 and 3','2 and 3'};
for i = 1:3
    figure
    plot([1:1000:2*10^5],norm_cost(i,:));
    xlabel('Iteration');
    ylabel('Normalized Cost');
    title(['Normalized Cost vs #iteration for class ', class{i}]);
end
%% b
for i = 1:3
    figure
    plot([1:1000:2*10^5],ccr_train(i,:));
    xlabel('Iteration');
    ylabel('Training CCR');
    title(['Training CCR vs #iteration for class ', class{i}]);
end

%% c
for i = 1:3
    figure
    plot([1:1000:2*10^5],ccr_test(i,:));
    xlabel('Iteration');
    ylabel('Test CCR');
    title(['Test CCR vs #iteration for class ', class{i}]);
end

%% d
for i = 1:3
    fprintf('Theta of classes %s\n',class{i});
    disp(theta(:,i));
    fprintf('Training CCR of classes %s ',class{i});
    disp(ccr_train(i,end));
    fprintf('Test CCR of classes %s ',class{i});
    disp(ccr_test(i,end));
    fprintf('Training Confusion Matrix of classes %s\n',class{i});
    disp(confusionmat(ypred_train(:,i),y_train));
    fprintf('Test Confusion Matrix of classes %s\n',class{i});
    disp(confusionmat(ypred_test(:,i),y_test));
end

%% e
train_AP = zeros(n_train,1);
ccr_AP = 0;
for n = 1:n_train
    win = zeros(1,m);
    ypredAP = sign(theta'*[X_train(n,:),1]');
    if ypredAP(1) > 0
        win(1) = win(1)+1;
    elseif ypredAP(1) < 0
        win(2) = win(2)+1;
    end
    if ypredAP(2) > 0
        win(1) = win(1)+1;
    elseif ypredAP(2) < 0
        win(3) = win(3)+1;
    end
    if ypredAP(3) > 0
        win(2) = win(2)+1;
    elseif ypredAP(3) < 0
        win(3) = win(3)+1;
    end
    [~,i] = max(win);
    train_AP(n) = i;
    if i == Y_label_train(n)
        ccr_AP = ccr_AP + 1;
    end
end
ccr_AP = ccr_AP/n_train;

test_AP = zeros(n_test,1);
ccr_testAP = 0;
for n = 1:n_test
    win = zeros(1,m);
    ypredAP = sign(theta'*[X_test(n,:),1]');
    if ypredAP(1) > 0
        win(1) = win(1)+1;
    elseif ypredAP(1) < 0
        win(2) = win(2)+1;
    end
    if ypredAP(2) > 0
        win(1) = win(1)+1;
    elseif ypredAP(2) < 0
        win(3) = win(3)+1;
    end
    if ypredAP(3) > 0
        win(2) = win(2)+1;
    elseif ypredAP(3) < 0
        win(3) = win(3)+1;
    end
    [~,i] = max(win);
    test_AP(n) = i;
    if i == Y_label_test(n)
        ccr_testAP = ccr_testAP + 1;
    end
end
ccr_testAP = ccr_testAP/n_test;

fprintf('For AP Method\n');
fprintf('Training CCR\n');
disp(ccr_AP);
fprintf('Test CCR\n');
disp(ccr_testAP);
fprintf('Training Confusion Matrix\n');
disp(confusionmat(Y_label_train,train_AP));
fprintf('Test Confusion Matrix\n');
disp(confusionmat(Y_label_test,test_AP));