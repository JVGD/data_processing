%% Fco Javier Vargas Garcia-Donas
% P3 - Lineal Regression

% Loading data
% close all;
% clear all;
load('datos_energia_eolica_P1.mat');

% Getting parameters from our data
[N,M] = size(X);
x = X(:,1);        % Wind speed

% Discarting non-relevant data and adding indepent component

    % Simple rule TODO: change it
    % Getting rid of the outliers
    index = x < 20;
    
    % Updating our data
    x = x(index);
    Y = Y(index);
    
% Model 2: Solving by normal equations and regularization
    
    % Building the 3 order system
    X = [ones(length(x),1)  x  x.^2  x.^3];
    
    % Getting new parameters
    [N,M] = size(X);
    
    % Weights & regularization
    lambda = 1.5;
    lambda = [0, lambda * ones(1,M-1)];
    lambda_matrix = diag(lambda);
    
% Fitting the model
    
    % Normal equation with lambda regularization
    w = inv(X' * X + lambda_matrix) * X' * Y;
    
% Predictions

    % Preprocessing test data
    xtt = Xtt(:,1);
    Xtt = [ones(length(xtt),1)  xtt xtt.^2  xtt.^3];

    % Model: Y = X * w
    Y_train = X * w;
    Y_est = Xtt * w;

% Plotting

    figure();
    plot(X(:,2), Y,'.');                % Train data
    hold on;
    plot(X(:,2), Y_train, '.r');        % Fitted model
    hold on;
    plot(Xtt(:,2), Ytt, '.m');          % Test data

    title('Prediction with Model 2: Normal Equations & Regularization');
    xlabel('Wind Speed [m/s]');
    ylabel('Energy [KWh]');
    legend('Train data', 'Fitted model', 'Test data');
    axis([min(X(:,2)) max(X(:,2)) min(Y) max(Y)]);

% Evaluation of the regressor

    % Computing the EV (explained variance)
    Ns = length(Ytt);
    sigma2 = var(Ytt);
    
    mse = 1/(Ns) * sum( (Y_est - Ytt).^2 );
    ev = 1 - mse/sigma2;
    fprintf('EV error of model 2:    %f\n', ev);

    
    
    
