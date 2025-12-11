% ================================================================
% Simulation of Ossa vs LM estimation under different scenarios
% ================================================================
clear all; 
clc; ß
close all;

rng(42);

addpath(genpath('../01_data'));
addpath(genpath('../02_ossa_original'));
addpath(genpath('../03_lm'));

% Load Data
DATA = load('../01_data/DATA.mat');
TARIFF = DATA.TARIFF;
TARGETTARIFF = DATA.TARGETTARIFF;

N = size(TARIFF, 2);
S = size(TARIFF, 1) / N;

TEMP = reshape(TARIFF', [N, N, S]);
TARIFFs = permute(TEMP, [2, 1, 3]);

% ================================================================
% Original data: country tariff variance
fprintf('Computing original tariff variance by country...\n');
country_tariff_var = zeros(N, 2);
for j = 1:N
    country_tariff_var(j, :) = [j, var(TARGETTARIFF(:, j))];
end

country_var_table = array2table(country_tariff_var, 'VariableNames', {'Country', 'Tariff_Var'});
writetable(country_var_table, '../04_simulations/original_tariff_variance.csv');
fprintf('Saved: original_tariff_variance.csv\n\n');

% ================================================================
% Setup
target_country = 1;
num_trials = 50;

TARGETTARIFF_original = TARGETTARIFF(:, target_country);
UB = max(TARGETTARIFF_original + 0.03, 2.25);
MFN_guess = mean(TARGETTARIFF_original) * ones(S, 1);

% Get baseline Ossa lambda
fprintf('Computing baseline Ossa lambda for Country %d...\n', target_country);
evalc('[lambda_ossa_baseline, ~] = mylambdaj(target_country, TARGETTARIFF_original);');
fprintf('Done\n\n');

% ================================================================
% Scenario 1: Realistic
fprintf('Scenario 1: Realistic structure\n');

results_s1 = zeros(num_trials, 4); 
% [lambda_var, tariff_var, norm_ossa, norm_lm]

tic;
for trial = 1:num_trials
    fprintf('Trial %d/%d\n', trial, num_trials);
    
    noise_level = 0.05;
    lambda_true = lambda_ossa_baseline .* (1 + noise_level * randn(S, 1));
    lambda_true = max(lambda_true, 1e-6);
    lambda_true = lambda_true / mean(lambda_true);
    
    lambda_var = var(lambda_true);
    
    LAMBDA_true = ones(N, S);
    LAMBDA_true(target_country, :) = lambda_true';
    target_tariff_new = mymfnoptimaltariffj(target_country, TARIFFs, LAMBDA_true, 0, UB, MFN_guess);
    TARGETTARIFF(:, target_country) = target_tariff_new;
    
    tariff_var = var(target_tariff_new);
    
    evalc('[lambda_ossa, ~] = mylambdaj(target_country, target_tariff_new);');
    norm_ossa = norm(lambda_ossa - lambda_true);
    
    [lambda_lm, ~] = estimate_lambda_lm(target_country, TARIFFs, TARGETTARIFF);
    norm_lm = norm(lambda_lm - lambda_true);
    
    results_s1(trial, :) = [lambda_var, tariff_var, norm_ossa, norm_lm];
end
time_s1 = toc;

fprintf('Scenario 1 completed in %.1f seconds\n', time_s1);
fprintf('Ossa: %.4f, LM: %.4f\n\n', mean(results_s1(:,3)), mean(results_s1(:,4)));

s1_table = array2table(results_s1, 'VariableNames', {'Lambda_Var', 'Tariff_Var', 'Norm_Ossa', 'Norm_LM'});
writetable(s1_table, '../04_simulations/scenario1_realistic.csv');

% ================================================================
% Scenario 2: Unrealistic
fprintf('Scenario 2: Unrealistic structure\n');

results_s2 = zeros(num_trials, 4);

tic;
for trial = 1:num_trials
    fprintf('Trial %d/%d\n', trial, num_trials);
    
    lambda_true = 0.5 + 9 * rand(S, 1);
    lambda_true = lambda_true / mean(lambda_true);
    
    lambda_var = var(lambda_true);
    
    LAMBDA_true = ones(N, S);
    LAMBDA_true(target_country, :) = lambda_true';
    target_tariff_new = mymfnoptimaltariffj(target_country, TARIFFs, LAMBDA_true, 0, UB, MFN_guess);
    TARGETTARIFF(:, target_country) = target_tariff_new;
    
    tariff_var = var(target_tariff_new);
    
    evalc('[lambda_ossa, ~] = mylambdaj(target_country, target_tariff_new);');
    norm_ossa = norm(lambda_ossa - lambda_true);
    
    [lambda_lm, ~] = estimate_lambda_lm(target_country, TARIFFs, TARGETTARIFF);
    norm_lm = norm(lambda_lm - lambda_true);
    
    results_s2(trial, :) = [lambda_var, tariff_var, norm_ossa, norm_lm];
end
time_s2 = toc;

fprintf('Scenario 2 completed in %.1f seconds\n', time_s2);
fprintf('Ossa: %.4f, LM: %.4f\n\n', mean(results_s2(:,3)), mean(results_s2(:,4)));

s2_table = array2table(results_s2, 'VariableNames', {'Lambda_Var', 'Tariff_Var', 'Norm_Ossa', 'Norm_LM'});
writetable(s2_table, '../04_simulations/scenario2_unrealistic.csv');

fprintf('Done\n');