%% Fco Javier Vargas Garcia-Donas
% P3 - Lineal Regression
% Data inspection

% Loading data
close all;
clear all;
load('datos_energia_eolica_P1.mat');

% Naming our data
N = size(X,1);
M = size(X,2);
vel = X(:,1);
dir = X(:,2);
ene = Y;

% Taking a look at our data

    % Wind Speed and Energy
    figure
    plot(vel,ene,'.');
    title('Energy( wind speed )');
    xlabel('Wind Speed [m/s]');
    ylabel('Energy [KWh]');

    % Wind Direction and Energy
    figure
    plot(dir,ene,'.');
    title('Energy( wind direction )');
    xlabel('Wind Direction [Aº]');
    ylabel('Energy [KWh]');

    % Transforming the direction
    vel_x = vel .* cos(dir);
    vel_y = vel .* sin(dir);

    % Ploting wind direcction decomposed
    figure;
    scatter3(vel_x, vel_y, ene);
    axis('square');
    title('Wind Direction');
    xlabel('Wind Direction X component [m/s]');
    ylabel('Wind Direction Y component [m/s]');
    zlabel('Energy [KWh]');

    fprintf('It is not dependent on wind direction\n');

% Discarting non-relevant data and adding indepent component

    % we change to variable_p (variable processed)
    index = vel < 20;
    vel_p = vel(index);
    ene_p = ene(index);
    Np = length(vel_p);
    
    figure
    plot(vel_p,ene_p,'.');
    title('Energy with filtered outliers');
    xlabel('Wind Speed [m/s]');
    ylabel('Energy [KWh]');
    
    % Adding the independent component
    Xp = [ones(Np,1), vel_p];
    yp = ene_p;
    

    
    
    
    
    
