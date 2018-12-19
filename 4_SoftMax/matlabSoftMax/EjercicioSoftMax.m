% % Alumn: Javier Vargas: Fco Javier Vargas Garcia-Donas
% Provide only the label with any prob (the prob is not as important)

% Add path for custom
% minimizer
clear all,close all
restoredefaultpath
addpath ./common/

% Based on the material at
% http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

% Load the MNIST data for this exercise.
% train.X and test.X will contain the training and testing images.
%   Each matrix has size [n,m] where:
%      m is the number of examples.
%      n is the number of pixels in each image.
% train.y and test.y will contain the corresponding labels (1 to 10).

% Previous info
binary_digits = false;      % If we just one to classify 0 and 1
num_classes = 10;

% Loading the data
addpath data                % where unzipped data is
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
test.X  = [ones(1,size(test.X,2)); test.X];
train.y = train.y + 1;      % make labels 1-based.
test.y  = test.y + 1;       % make labels 1-based.

train.X = train.X';
test.X  = test.X';
train.y = train.y';
test.y  = test.y';

% Training set dimensions
n = size(train.X,2);
m = size(train.X,1);        % 28 * 28 pixels = 785 pixels = 785 weights

% Train softmax classifier using minFunc
options = struct('MaxIter', 100);

% Initialize theta.  
% We use a matrix where each column corresponds to a class,
% and each row is a classifier coefficient for that class.
% Inside minFunc, theta will be stretched out into a long 
% vector (theta(:)).
% We only use num_classes-1 columns, since the last column is 
% always assumed 0.
theta = rand(n,num_classes-1)*0.001;


tic;

%% TO DO
% Write function softmax_regression_vec(theta, train.X, train.y) that
% returns the value of the cost function and its derivative  with respect
% to the every entry of the weights.
% It is of the form:
% function [f,g] = softmax_regression_vec(w, X,y)
% w is a column vector with the weights, of length the number of labels 
% minus one, Q-1, by the number of inputs plus ones. It has (Q -1) x (n+1) dimensions.
% The gradient, g, has the same dimensions.


%% MATLAB Optimization Toolbox, very consuming, do not try
% options = optimoptions('fminunc','GradObj','on','Display','iter','Algorithm','trust-region','MaxIter',3);
% [theta,fval] = fminunc(@(theta) @softmax_regression_vec(theta, train.X, train.y), theta ,options);


%% Standford minFunc
addpath common/minFunc_2012/minFunc
addpath common/minFunc_2012/minFunc/compiled
theta(:)=minFunc(@softmax_regression_vec, theta(:), options, train.X, train.y);

fprintf('Optimization took %f seconds.\n', toc);
theta=[theta, zeros(n,1)]; % expand theta to include the last class.

% Print out training accuracy.
X = train.X;
w = theta;
XW = X*w;                           % matrix of x(n) w(q)
eXW = exp(XW);                      % exponential of the matrix
reg = sum(eXW,2);                   % regularization term
reg = repmat(reg,1,num_classes);    % for .* multiplication
hw_x = 1./reg .* eXW;               % probs for each class in rows
                                    % NOTE: we could have omited the reg

% The label will be the one with 
% the max prob that is the same 
% as the index check lines 31,32
[~,labels] =  max( hw_x.' );        % transposing for max per column
labels = labels.';                  % back to columns

% Note that we only ask for the hard classification, i.e., the label 
% with highest probability
correct=sum(train.y == labels);
accuracy = correct / length(train.y);

fprintf('\nTraining accuracy: %2.1f%%\n', 100*accuracy);

% Print out test accuracy.
X = test.X;
w = theta;
XW = X*w;                           % matrix of x(n) w(q)
eXW = exp(XW);                      % exponential of the matrix
reg = sum(eXW,2);                   % regularization term
reg = repmat(reg,1,num_classes);    % for .* multiplication
hw_x = 1./reg .* eXW;               % probs for each class in row

[~,labels] =  max( hw_x.' );        % transposing for max per column
labels = labels.';                  % back to columns

correct = sum(test.y == labels);
accuracy = correct / length(test.y);

fprintf('Test accuracy: %2.1f%%\n', 100*accuracy);

wrong = find( test.y ~= labels );

fprintf('\nRun Check Wrongs for more insights on\n');
fprintf('how the algorithm fails\n');
pause();



