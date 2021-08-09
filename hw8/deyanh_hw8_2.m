%% load data
clear,clc;
load('kernel-svm-2rings.mat')
n = length(x);
rng('default');
%% algorithm
K = exp(-(1/(2*0.5^2))*(diag(x'*x)+ diag(x'*x)' - 2*x'*x));
K_ext = [K;ones(1,n)];
phi = zeros(n+1,1);
cost=zeros(100,1);
ccr = zeros(100,1);

for t = 1:1000
    j = randi([1,n]);
    v = [K zeros(n,1);zeros(1,n+1)]*phi;
    if y(j) * phi' * K_ext(:,j) < 1
        v = v - 256 * y(j) *  K_ext(:,j);
    end
    phi = phi - (0.256/t)*v;
    
    if mod(t,10) == 0
        g = (1/2)*phi'*[K zeros(n,1);zeros(1,n+1)]*phi+sum(256/n*max(0,1-y'.* (phi'*K_ext)));
        c = sum(sign(phi'* K_ext)==y');
        cost(t/10)=g/n;
        ccr(t/10)=c/n;
    end
end
   
%% 8.2a
figure
plot([1:10:1000],cost);
xlabel('Iteration Number');
ylabel('Normalized Cost');
title('Normalized Cost vs. #Iteration');
ylim([0 100]);

%% 8.2b
figure
plot([1:10:1000],ccr);
xlabel('Iteration Number');
ylabel('Training CCR');
title('Training CCR vs. #Iteration');

%% 8.2c
fprintf('Training Confusion Matrix: \n');
disp(confusionmat(y,sign(phi'* K_ext)));

%% 8.2d
[Xgrid, Ygrid]=meshgrid([-2:.1:2],[-2:.1:2]);
Xtest = [Xgrid(:),Ygrid(:)];
Ktest=exp(-(1/(2*0.5^2))*(diag(Xtest*Xtest')'+diag(x'*x)- 2*x'*Xtest'));
Ktest_ext = [Ktest;ones(1,length(Ktest))];
pred = sign(phi'*Ktest_ext);

figure
gscatter(x(1,:),x(2,:),y(:),'rgb');
hold on
gscatter(Xgrid(:),Ygrid(:),pred,'rgb','+');
hold off
xlabel('Feature 1');
ylabel('Feature 2');
title('Training data & Decision Boundary');
legend('Class 1','Class 2');

