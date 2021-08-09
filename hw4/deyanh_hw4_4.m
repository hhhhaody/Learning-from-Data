%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503 (Ishwar) Fall 2020
% HW 4
% Deyan Hao (deyanh@bu.edu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;
rng('default')  % For reproducibility of data and results

load("prostateStnd.mat")

%% a.Normalization
xmean = mean(Xtrain);
xstd = std(Xtrain,1);
ymean = mean(ytrain);
ystd = std(ytrain,1);
fprintf('4.4a\n');
fprintf('means of features before normalization %f \n', mean(Xtrain));
fprintf('variance of features before normalization %f \n', var(Xtrain));
fprintf('means of label before normalization %f \n', mean(ytrain));
fprintf('variance of label before normalization %f \n', var(ytrain));
normxtrain = (Xtrain-xmean)./xstd;
normytrain = (ytrain-ymean)./ystd;
normxtest = (Xtest-xmean)./xstd;
normytest = (ytest-ymean)./ystd;
%% 4.4b. Ridge Regression
lambda = exp(-5:10);
w_ridge = zeros(length(lambda),8);
b_ridge = zeros(length(lambda),1);
for i = 1:length(lambda)
    sx = 1/length(normxtrain)*(normxtrain'*normxtrain);
    sxy = 1/length(normxtrain)*normxtrain'*normytrain;
    w_ridge(i,:) = (lambda(i) / length(normxtrain) * eye(8) + sx)^-1 * sxy;
    b_ridge(i) = mean(normytrain) - w_ridge(i,:)* mean(normxtrain)';
end
%% 4.4c.
e = -5:10;
figure
for i = 1:8
    plot(e,w_ridge(:,i));
    hold on
end
hold off
xlabel('ln \lambda');
ylabel('Ridge Regression Coefficient');
title('Ridge Regression Coefficient vs ln \lambda');
legend(names(1:8));
%% 4.4d
MSE_train = zeros(length(lambda),1);
MSE_test = zeros(length(lambda),1);
for i = 1:length(lambda)
    for j = 1:length(normxtrain)
        MSE_train(i) = MSE_train(i) + (normytrain(j)- w_ridge(i,:)*normxtrain(j,:)')^2;
    end
    MSE_train(i) = MSE_train(i)/length(normxtrain);
    
    for j = 1:length(normxtest)
        MSE_test(i) = MSE_test(i) + (normytest(j)- w_ridge(i,:)*normxtest(j,:)')^2;
    end
    MSE_test(i) = MSE_test(i)/length(normxtest);
    
end
figure
plot(e,MSE_train);
hold on
plot(e,MSE_test);
hold off
xlabel('ln \lambda');
ylabel('MSE');
title('Mean-Squared-Error of Training and Test sets versus ln \lambda');
legend('MSE train','MSE test');