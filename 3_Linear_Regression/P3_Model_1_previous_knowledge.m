%% Fco Javier Vargas Garcia-Donas
% P3 - Lineal Regression

% Loading data
close all;
clear all;
load('datos_energia_eolica_P1.mat');

% Getting parameters from our data
N = size(X,1);
M = size(X,2);
x = X(:,1);        % Wind speed

% Discarting non-relevant data and adding indepent component

    % Simple rule TODO: change it
    % Getting rid of the outliers
    index = x < 20;
    
    % Updating our data
    x = x(index);
    Y = Y(index);
    
% Model 1: Solving based on prior knowledge of the problem

    % Getting the X matrix for regression
	X = [x];

    % Getting new parameters
    [N,M] = size(X);
    
 % Fitting the model
 
    C = 1/N * sum( Y ./ X.^3 );
    
% Predictions
    
    % Preprocessing test data
    Xtt = Xtt(:,1);             % We use only wind speed

    % Model: y = C * v^3
    Y_train = C * X.^3;
    Y_est  = C * Xtt.^3;

% Plotting    
    
    figure();
    plot(X, Y, '.');            % Plotting the data
    hold on;
    plot(X, Y_train, '.r');     % Plotting fitted model
    hold on;
    plot(Xtt, Ytt, '.m');       % Plotting test data
    hold off;
    
    title('Prediction with Model 1: y = C v^3');
    xlabel('Wind Speed [m/s]');
    ylabel('Energy [KWh]');
    legend('Train data', 'Fitted model', 'Test data');
    axis([min(X) max(X) min(Y) max(Y)]);
    
% Evaluation of the regressor

    % Computing the EV (explained variance)
    Ns = length(Ytt);
    sigma2 = var(Ytt);
    
    mse = 1/(Ns) * sum( (Y_est - Ytt).^2 );
    ev = 1 - mse/sigma2;
    fprintf('EV error of model 1:    %f\n', ev);    
    

    
    
    
