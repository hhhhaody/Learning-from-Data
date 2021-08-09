function [lambda_top5, k_] = deyanh_hw5_2()
%% Q5.2
%% Load AT&T Face dataset
    img_size = [112,92];   % image size (rows,columns)
    % Load the AT&T Face data set using load_faces()
    %%%%% TODO
    X = load_faces();
    [n,d]= size(X);
    
    %% Compute mean face and the covariance matrix of faces
    % compute X_tilde
    %%%%% TODO
    mu = mean(X);
    X_tilde = X - mu;
    % Compute covariance matrix using X_tilde
    %%%%% TODO
    S_x = X_tilde'*X_tilde/n;
    %% Compute the eigenvalue decomposition of the covariance matrix
    %%%%% TODO
    [eigenvectors, eigenvalues] = eig(S_x);
    %% Sort the eigenvalues and their corresponding eigenvectors construct the U and Lambda matrices
    %%%%% TODO
    lambda = diag(eigenvalues);
    [lambda,index]=sort(lambda,'descend');
    U = zeros(d,d);
    for i = 1:d
        U(:,i)=eigenvectors(:,index(i));
    end
    %% Compute the principal components: Y
    %%%%% TODO
    Y = U'*X_tilde';
%% Q5.2 a) Visualize the loaded images and the mean face image
    figure(1)
    sgtitle('Data Visualization')
    
    % Visualize image number 120 in the dataset
    % practice using subplots for later parts
    subplot(1,2,1)
    %%%%% TODO
    imshow(uint8(reshape(X(120,:), img_size)));
    title('the 120th face image');
    % Visualize the mean face image
    subplot(1,2,2)
    imshow(uint8(reshape(mu, img_size)));
    title('the mean face image');
    %%%%% TODO
    
%% Q5.2 b) Analysing computed eigenvalues
    warning('off')
    
    % Report the top 5 eigenvalues
    lambda_top5 = lambda(1:5);
    fprintf('5.2b: first 5 eigenvalues:\n');
    lambda_top5
    
    % Plot the eigenvalues in from largest to smallest
    k = 1:d;
    figure(2)
    sgtitle('Eigenvalues from largest to smallest')

    % Plot the eigenvalue number k against k
    subplot(1,2,1)
    %%%%% TODO
    plot(k(1:450),lambda(1:450));
    title('top 450 lambdas');
    xlabel('K');
    ylabel('Lambdas');
    
    % Plot the sum of top k eigenvalues, expressed as a fraction of the sum of all eigenvalues, against k
    %%%%% TODO: Compute eigen fractions
    eigenfraction = zeros(450,1);
    for i= 1:450
        eigenfraction(i) = round(sum(lambda(1:i))/sum(lambda),2);
    end
        
    subplot(1,2,2)
    %%%%% TODO
    plot(k(1:450),eigenfraction);
    title('eigen fractions');
    xlabel('K');
    ylabel('eigen fractions');
    
    % find & report k for which the eigen fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    %%%%% TODO (Hint: ismember())
    [~,index]=ismember(ef,eigenfraction);
    k_ = index;
    fprintf('k eigenvalues for eigen fractions\n');
    disp(k_);
    
%% Q5.2 c) Approximating an image using eigen faces
    test_img_idx = 43;
    test_img = X(test_img_idx,:);    
    % Compute eigenface coefficients
    %%%% TODO
    
    K = [0,1,2,k_,d];
    % add eigen faces weighted by eigen face coefficients to the mean face
    % for each K value
    % 0 corresponds to adding nothing to the mean face
    x_bars = zeros(d,length(K));
    for i = 1:length(K)
        x_bars(:,i) = mu' + U(:,1:K(i))*Y(1:K(i),test_img_idx);
    end

    % visulize and plot in a single figure using subplots the resulating image approximations obtained by adding eigen faces to the mean face.

    %%%% TODO 
    
    figure(3)
    sgtitle('Approximating original image by adding eigen faces')
    for i = 1:length(K)
        subplot(3,3,i);
        imshow(uint8(reshape((x_bars(:,i)'), img_size)));
        title(sprintf('face for k = %d',(K(i))));
    end
%% Q5.2 d) Principal components capture different image characteristics
%% Loading and pre-processing MNIST Data-set
    % Data Prameters
    q = 5;                  % number of quantile points
    noi = 3;                % Number of interest
    img_size = [16, 16];
    
    % load mnist into workspace
    mnist = load('mnist256.mat').mnist;
    label = mnist(:,1);
    X = mnist(:,(2:end));
    num_idx = (label == noi);
    X = X(num_idx,:);
    [n,d] = size(X);
    
    %% Compute the mean face and the covariance matrix
    % compute X_tilde
    %%%%% TODO
    mu = mean(X);
    X_tilde = X-mu;
   
    % Compute covariance using X_tilde
    %%%%% TODO
    S_x = X_tilde'*X_tilde/n;
    
    %% Compute the eigenvalue decomposition
    %%%%% TODO
    [eigenvectors, eigenvalues] = eig(S_x);
    
    %% Sort the eigenvalues and their corresponding eigenvectors in the order of decreasing eigenvalues.
    %%%%% TODO
    lambda = diag(eigenvalues);
    [lambda,index]=sort(lambda,'descend');
    U = zeros(d,d);
    for i = 1:d
        U(:,i)=eigenvectors(:,index(i));
    end
    
    %% Compute principal components
    %%%%% TODO
    Y = U'*X_tilde';
    %% Computing the first 2 pricipal components
    %%%%% TODO
    y1 = Y(1,:)';
    y2 = Y(2,:)';
    % finding quantile points
    quantile_vals = [0.05, .25, .5, .75, .95];
    %%%%% TODO (Hint: Use the provided fucntion - quantile_points())
    percentilepoints1 = percentile_values(y1,quantile_vals*100);
    percentilepoints2 = percentile_values(y2,quantile_vals*100);
    % Finding the cartesian product of quantile points to find grid corners
    %%%%% TODO
    [Xgrid, Ygrid]=meshgrid(percentilepoints1,percentilepoints2);
    corners = [Xgrid(:) Ygrid(:)];
    %% Find images whose PCA coordinates are closest to the grid coordinates 
    
    %%%%% TODO
    pc = [y1 y2];
    distance = diag(corners*corners')+ diag(pc*pc')' - 2*corners*pc';
    [~,nearest]=min(distance,[],2);
    closest=pc(nearest,:);
        
    %% Visualize loaded images
    % random image in dataset
    figure(4)
    sgtitle('Data Visualization')

    % Visualize the 120th image
    subplot(1,2,1)
    %%%%% TODO
    imshow(reshape(X(120,:), img_size));
    title('the 120th image');
    % Mean face image
    subplot(1,2,2)
    %%%%% TODO
    imshow(reshape(mu, img_size));
    title('the mean image');
    
    %% Image projections onto principal components and their corresponding features
    
    figure(5)    
    hold on
    grid on
    
    % Plotting the principal component 1 vs principal component 2. Draw the
    % grid formed by the quantile points and highlight the image points that are closest to the 
    % quantile grid corners
    
    %%%%% TODO (hint: Use xticks and yticks)
    
    scatter(y1,y2);
    scatter(closest(:,1),closest(:,2),'r','filled');
    xticks(percentilepoints1);
    xticklabels({'5^t^h','25^t^h','50^t^h','75^t^h','95^t^h'})
    yticks(percentilepoints2);
    yticklabels({'5^t^h','25^t^h','50^t^h','75^t^h','95^t^h'})

    xlabel('Principal component 1')
    ylabel('Principal component 2')
    title('Image points closest to quantile grid corners')
    hold off
    
    figure(6)
    sgtitle('Images closest to quantile grid corners')
    hold on
    % Plot the images whose PCA coordinates are closest to the quantile grid 
    % corners. Use subplot to put all images in a single figure in a grid.
    
    %%%%% TODO
    p = repmat(quantile_vals*100,5,1);
    p1=p';
    percentiles = [p(:) p1(:)];
    for i = 1:25
        subplot(5,5,i)
        imshow((reshape(X(nearest(i), :), img_size)));
        title([num2str(percentiles(i,1)),'% of pc1 ',num2str(percentiles(i,2)),'% of pc2']);
    end
    hold off    
end