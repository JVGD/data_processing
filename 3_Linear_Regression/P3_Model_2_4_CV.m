%% Fco Javier Vargas Garcia-Donas
% P3 - Lineal Regression

% Loading data
close all;
clear all;
load('datos_energia_eolica_P1.mat');

% Getting the wind speed only
x = X(:,1);
xtt = Xtt(:,1);

% Discarting non-relevant data and adding indepent component

    % Getting rid of the outliers
    index = x < 20;
    x = x(index);
    Y = Y(index);

    % Same for test data
    index = xtt < 20;
    xtt = xtt(index);
    Ytt = Ytt(index);
    
% Joining all data together for Cross Validation

    X = [x ; xtt];        % We only consider wind speed
    Y = [Y ; Ytt];

% Leaving 20% for final tests
    
    test_len = floor( 0.2 * length(X) );
    x_len = length(X);

    Xtt = X(x_len-test_len:end);    
    Ytt = Y(x_len-test_len:end);    
    X   = X(1:x_len-test_len-1); 
    Y   = Y(1:x_len-test_len-1);

    % Getting model params
    [N,M] = size(X);
    
% Preparing loop
L = 10;                 % Num Iterations
W = zeros(4, L);        % Storing weights of iterations
EV = zeros(1,L);        % Storing the error

for i = 1:L;    
    
    % We sample the train data for 
    %   - 70% train 
    %   - 30% cross validation

        % Vector from 1 to N permuted randomly
        perm_index = randperm(N);
        stop_index = floor( 0.7 * N );

        tr_index = perm_index(1:stop_index);
        cv_index = perm_index(stop_index+1:end);

        Xtr = X(tr_index);
        Xcv = X(cv_index);

        Ytr = Y(tr_index);
        Ycv = Y(cv_index);

    % Model 2: Solving by normal equations and regularization

        % Building the 3 order system
        Xtr = [ones(length(Xtr),1)  Xtr  Xtr.^2  Xtr.^3];

        % Getting new parameters
        [N,M] = size(Xtr);

        % Weights & regularization
        lambda = 1.5;
        lambda = [0, lambda * ones(1,M-1)];
        lambda_matrix = diag(lambda);

    % Fitting the model

        % Normal equation with lambda regularization
        w = inv(Xtr' * Xtr + lambda_matrix) * Xtr' * Ytr;

    % Predictions

        % Building model for CV datas
        Xcv = [ones(length(Xcv),1)  Xcv Xcv.^2  Xcv.^3];

        % Prediction Model: Y = X * w
        Ycv_est = Xcv * w;

    % Computing the EV (explained variance)

        Ns = length(Ycv_est);
        sigma2 = var(Ycv_est);

        mse = 1/(Ns) * sum( (Ycv_est - Ycv).^2 );
        ev = 1 - mse/sigma2;
        fprintf('EV error of CV iteration %d:    %f\n', i, ev);
        
	% Storing the weights and the EV error
        W(:,i) = w;
        EV(i) = ev;
    
end

% Getting W as the average of the weights calculated
    w = mean(W,2);  

    % Predictions

        % Building model for test data
        Xtt = [ones(length(Xtt),1)  Xtt Xtt.^2  Xtt.^3];

        % Prediction Model: Y = X * w
        Ytt_est = Xtt * w;

    % Computing the EV (explained variance)

        Ns = length(Ytt_est);
        sigma2 = var(Ytt_est);

        mse = 1/(Ns) * sum( (Ytt_est - Ytt).^2 );
        ev = 1 - mse/sigma2;
        fprintf('\nEV error of the test data when averaging:    %f\n', ev);

        
% Getting weights as the one that get better EV error
    [max_EV, index] = max(EV);
    w = W(:,index);  

    % Predictions

        % Prediction Model: Y = X * w
        Ytt_est = Xtt * w;

    % Computing the EV (explained variance)

        Ns = length(Ytt_est);
        sigma2 = var(Ytt_est);

        mse = 1/(Ns) * sum( (Ytt_est - Ytt).^2 );
        ev = 1 - mse/sigma2;
        fprintf('EV error of the test data for best EV   :    %f\n', ev);
        
        
% Getting weights as the average of the best 3 (better EV error)
    [max_EV, index] = sort(EV, 'descend');
    w = W(:,index(1:3));
    w = mean(w,2);

    % Predictions

        % Prediction Model: Y = X * w
        Ytt_est = Xtt * w;

    % Computing the EV (explained variance)

        Ns = length(Ytt_est);
        sigma2 = var(Ytt_est);

        mse = 1/(Ns) * sum( (Ytt_est - Ytt).^2 );
        ev = 1 - mse/sigma2;
        fprintf('EV error of the test data for 3 best EV :    %f\n', ev);        





