%% 6.3 Load data
clear;clc;
load('iris.mat')
rng('default')
n_train = length(X_data_train);
n_test = length(X_data_test);
d = length(X_data_train(1,:));
m = 3;
%% 6.3a Histograph
Y = [Y_label_train;Y_label_test];
figure
histogram(Y);
xlabel('Classes');
ylabel('Counts');
title('Histogram of Labels');
xticks([1 2 3]);
%% 6.3a Correlation matrix
X = [X_data_train;X_data_test];
corr = cov(X)./sqrt(var(X)'*var(X));
fprintf('Correlation matrix\n');
disp(corr);
%% 6.3a Scatter plots of distinct features
p = 1;
figure
sgtitle('All distinct pairs of features');
hold on
for i = 1:3
    for j = (i+1):4
        subplot(3,2,p)
        gscatter(X(:,i),X(:,j),Y);
        xlim([0,8]);
        ylim([0,8]);
        xlabel(['Feature ',num2str(i)]);
        ylabel(['Feature ',num2str(j)]);
        title(['Scatter for feature ',num2str(i),' , ',num2str(j)]);
        p = p + 1;
    end
end
hold off
%% SGD

theta = zeros(d+1,m);
x_test_ext = [X_data_test, ones(n_test,1)];
x_train_ext = [X_data_train, ones(n_train,1)];
ccr_test = zeros(300,1);
ccr_train = zeros(300,1);
g_theta = zeros(300,1);
logloss = zeros(300,1);
t_max = 6000;
lambda = 0.1;


for t = 1:t_max
    j = randi(n_train);
    p_k = zeros(m,1);
    v_k = zeros(d+1,m);
   
    for k = 1:m
        sigma = sum(exp(theta' * x_train_ext(j,:)'));
        p_k(k) = exp(theta(:,k)' * x_train_ext(j,:)')/sigma;
        
        if p_k(k) < 10^(-10)
            p_k(k) = 10^(-10);
        end
        v_k(:,k) = 2*lambda*theta(:,k) + n_train*(p_k(k)-(k == Y_label_train(j)))*x_train_ext(j,:)';
    end
    
    theta = theta-(0.01/t)*v_k;
    
    if mod(t,20) == 0
        
        ypred = zeros(n_train,1);
        label_train = zeros(n_train,1);
        
        f0 = lambda*sum(diag(theta'*theta));
        
        fj = 0;
        for i = 1:n_train
            term1=log(sum(exp(theta'*x_train_ext(i,:)')));
            term2 = theta'*x_train_ext(i,:)';
            term2 = term2(Y_label_train(i));
            fj = fj + (term1 - term2);
            
        end
        g_theta(t/20) = f0 + fj;
        
        [~,train_pred]=max(theta'*x_train_ext');
        ccr_train(t/20) = sum(train_pred'==Y_label_train)/n_train;
        
        [~,test_pred]=max(theta'*x_test_ext');
        ccr_test(t/20) = sum(test_pred'==Y_label_test)/n_test;
        
        p_test_sum=0;
        for i = 1:n_test
            sigma2 = sum(exp(theta' * x_test_ext(i,:)'));
            p_test = exp(theta(:,Y_label_test(i))' * x_test_ext(i,:)')/sigma2;
        
            if p_test < 10^(-10)
                p_test = 10^(-10);
            end
            p_test_sum=p_test_sum+log(p_test);
        end
        logloss(t/20)=-p_test_sum/n_test;
    end
end

%% 6.3b L2 regularized logistic loss plot
figure
plot([1:20:6000],(g_theta/n_train));
xlabel('Iteration Number');
ylabel('l2 Regularized Logistic Loss');
title('l2 Regularized Logistic Loss vs iteration');

%% 6.3c Training set CCR
figure
plot([1:20:6000],ccr_train);
xlabel('Iteration Number');
ylabel('CCR of Training Set');
title('Training Set CCR vs iteration');

%% 6.3d Test set CCR
figure
plot([1:20:6000],ccr_test);
xlabel('Iteration Number');
ylabel('CCR of Test Set');
title('Test Set CCR vs iteration');

%% 6.3e Log loss of test set
figure
plot([1:20:6000],logloss);
xlabel('Iteration Number');
ylabel('Log-Loss of Test Set');
title('Test Set Log-Loss vs iteration');

%% 6.3f Final values
fprintf('Theta of Training Set:\n');
disp(theta);
fprintf('Training CCR:\n');
disp(ccr_train(300));
fprintf('Test CCR:\n');
disp(ccr_test(300));
fprintf('Training Confusion Matrix:\n');
disp(confusionmat(train_pred,Y_label_train));
fprintf('Test Confusion Matrix:\n');
disp(confusionmat(test_pred,Y_label_test));

%% 6.3g Decision region
p = 1;
[Xgrid, Ygrid]=meshgrid([0:.2:8],[0:.2:8]);
X_plot = [Xgrid(:),Ygrid(:)];
figure
hold on
for i = 1:3
    for j = (i+1):4
        x_plot=zeros(length(X_plot),5);
        x_plot(:,5)=1;
        x_plot(:,i)=X_plot(:,1);
        x_plot(:,j)=X_plot(:,2);
        [~,pred]=max(theta'*x_plot');
        
        subplot(3,2,p)
        gscatter(Xgrid(:),Ygrid(:),pred,'rgb');
        xlim([0,8]);
        ylim([0,8]);
        hold on
        xlabel(['Feature ',num2str(i)]);
        ylabel(['Feature ',num2str(j)]);
        title(['Scatter for feature ',num2str(i),' , ',num2str(j)]);
        p = p + 1;
    end
end
hold off