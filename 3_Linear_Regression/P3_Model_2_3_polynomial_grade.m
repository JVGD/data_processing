%% Fco Javier Vargas Garcia-Donas
% P3 - Lineal Regression

% Loading data
close all;
clear all;
load('datos_energia_eolica_P1.mat');

% Getting parameters from our data
[N,M] = size(X);
x = X(:,1);        % Wind speed

% Discarting non-relevant data and adding indepent component

    % Getting rid of the outliers
    index = x < 20;
    
    % Updating our data
    x = x(index);
    Y = Y(index);
    
% Model 2: Solving by normal equations and regularization
% Bulding the X and Xtest model of order 1
    
    % Train data
    X = [ones(length(x),1) x x.^2];

    % Test Building model for test data too
    xtt = Xtt(:,1);
    
    % Getting rid of the outliers
    index = xtt < 20;
    xtt = xtt(index);
    Ytt = Ytt(index);    
    
    % Building the model
    Xtt = [ones(length(xtt),1) xtt xtt.^2];

poly_grade = 5;

for n = 3:poly_grade

    % Building the system 
        X = [X  x.^n];

        % Getting new parameters
        [N,M] = size(X);
        
        fprintf('Iteration = %d N = %d M = %d\n', n, N, M);

    % Preprocessing test data
        Xtt = [Xtt  xtt.^n];

    % Looping

        N_loop = 10000;
        error_v = zeros(1, N_loop);
        mse_v = zeros(1, N_loop);
        lambda_v = linspace(0, 100, N_loop);

        for i = 1:N_loop;

             % Weights & regularization
                lambda = lambda_v(i);
                lambda = [0, lambda * ones(1,M-1)];
                lambda_matrix = diag(lambda);

            % Fitting the model

                % Normal equation with lambda regularization
                w = inv(X' * X + lambda_matrix) * X' * Y;

            % Predictions

                % Model: Y = X * w
                Y_train = X * w;
                Y_est = Xtt * w;

            % Evaluation of the regressor

                % Computing the EV (explained variance)
                Ns = length(Ytt);
                sigma2 = var(Ytt);

                mse = 1/(Ns) * sum( (Y_est - Ytt).^2 );
                ev = 1 - mse/sigma2;

            % Saving the error
                mse_v(i) = mse;
                error_v(i) = ev;

        end    

    % Plotting lambda VS error figure
        
        figure(1);
        plot(lambda_v, mse_v);
        title('Mean Squared Error (MSE)');
        xlabel('\lambda');
        ylabel('MSE(\lambda)');
        hold on;

        figure(2);
        plot(lambda_v, error_v);
        title('Explained Variance (EV)');
        xlabel('\lambda');
        ylabel('EV(\lambda)');
        hold on;
    
end

% Building the legend
mylegend = [];
for n = 3:poly_grade
    mylegend = [mylegend; 'Grade ' num2str(n)];
end

figure(1)
legend(mylegend)

figure(2)
legend(mylegend)

    
    
    
    
    
