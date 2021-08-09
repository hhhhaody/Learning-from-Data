%% load data
clear,clc;
load('kernel-kmeans-2rings.mat')
n = length(data);
%% initialization
K = exp(-(1/(2*0.16))*(diag(data*data')+ diag(data*data')' - 2*data*data'));
k = 2;
rng('default');
r1 = rand(n,1);
r2 = rand(n,1);
u1 = r1./(norm(r1));
u2 = r2./(norm(r2));
u = [u1,u2];
e = eye(n);
%%
for t = 1:100
    y=[diag((e-u(:,1))'*K*(e-u(:,1))),diag((e-u(:,2))'*K*(e-u(:,2)))];
    [~,i]=min(y,[],2);
    labels=i';
    
    for l = 1:2
        u(:,l)= ones(n,1).*(i==l);
        if sum(i==l) > 0
            u(:,l)= u(:,l)/sum(i==l);
        end
    end
end

%% Plotting Results
figure
gscatter(data(:,1),data(:,2),i);
xlabel('Feature 1');
ylabel('Feature 2');
title('Scatter of Kernel K-means Clustering Result');
legend('Class 1','Class 2');