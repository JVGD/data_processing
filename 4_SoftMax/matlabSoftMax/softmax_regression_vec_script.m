% Previous steps

    clear all,close all
    restoredefaultpath
    addpath common/

    binary_digits = false;      % If we just one to classify 0 and 1
    num_classes = 10;

    % Loading the data
    addpath data                %where unzipped data is
    [train,test] = ex1_load_mnist(binary_digits);

    train.X = [ones(1,size(train.X,2)); train.X]; 
    test.X  = [ones(1,size(test.X,2)); test.X];
    train.y = train.y + 1;      % make labels 1-based.
    test.y  = test.y + 1;       % make labels 1-based.

    train.X = train.X';
    test.X  = test.X';
    train.y = train.y';
    test.y  = test.y';
    
    

    n = size(train.X,2);
    m = size(train.X,1);        % 28 * 28 pixels = 785 pixels = 785 w

    options = struct('MaxIter', 100);

    theta = rand(n,num_classes-1)*0.001;
    
    
% Doing the function
    
    W = theta(:);               % Vectorizing theta matrix
    X = train.X;
    y = train.y;
    
    [M,N] = size(X);
    W = reshape(W, N, []);
    Q = size(W,2);              % Num of classes
    
    XW = X*W;                   % matrix of x(n) w(q)
    
    tic;
    eXW = exp(XW);              % exponential of the matrix
    eXW = [eXW ones(M,1)];
    
    I = sub2ind(size(eXW), [1:M]', y);
    Jw = -1/M * sum( log(eXW(I) ./ sum(eXW,2)) );
    toc;
    
    % This is one way to solve it
    %     tic;
    %     Jw = 0;
    %     for m = 1:M;
    %         for q = 1:Q;
    %            % We use the minus "-" cause of the Jw = - sum(...)
    %             Jw = Jw - (Y(m) == q) * log ( eXW(m,q) / sum(eXW(m,:)) );
    %         end
    %     end
    %     toc;
    
    
    
    