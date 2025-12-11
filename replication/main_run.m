% ================================================================
% Main script: Replicating Political Economy Weights in Ossa (2014)
% ================================================================
clear all; 
clc; 
close all;

rng(42);

addpath(genpath('01_data'));
addpath(genpath('02_ossa_original'));
addpath(genpath('03_lm'));

% Load Data
DATA = load('01_data/DATA.mat');
TARIFF = DATA.TARIFF;
TARGETTARIFF = DATA.TARGETTARIFF;

N = size(TARIFF, 2);
S = size(TARIFF, 1) / N;

TEMP = reshape(TARIFF', [N, N, S]);
TARIFFs = permute(TEMP, [2, 1, 3]);

% ================================================================
% Estimate lambda using the original Ossa method (mylambdaj.m)
tic;
LAMBDA_OSSA = zeros(S, N);
OPTIMAL_TARIFF_OSSA = zeros(S, N);

for j = 1:N
    TARGETTARIFFj = TARGETTARIFF(:, j);
    [LAMBDAj, MFNOPTIMALTARIFFj] = mylambdaj(j, TARGETTARIFFj);
    
    LAMBDA_OSSA(:, j) = LAMBDAj;
    OPTIMAL_TARIFF_OSSA(:, j) = MFNOPTIMALTARIFFj;
end
time_ossa = toc;
fprintf('Completed original method in %.1f seconds.\n', time_ossa);

save('05_output/lambda_ossa.mat', 'LAMBDA_OSSA', 'OPTIMAL_TARIFF_OSSA');
writematrix(LAMBDA_OSSA', '05_output/lambda_ossa.csv');
writematrix(OPTIMAL_TARIFF_OSSA', '05_output/optimal_tariff_ossa.csv');

% ================================================================
% Estimate lambda using Levenberg-Marquardt algorithm (estimate_lambda_lm.m)
tic;
LAMBDA_LM = zeros(S, N);
OPTIMAL_TARIFF_LM = zeros(S, N);

for j = 1:N
    [lambda_j, tariff_j] = estimate_lambda_lm(j, TARIFFs, TARGETTARIFF);
    
    LAMBDA_LM(:, j) = lambda_j;
    OPTIMAL_TARIFF_LM(:, j) = tariff_j;
    
end
time_lm = toc;
fprintf('Completed LM method in %.1f seconds.\n', time_lm);

save('05_output/lambda_lm.mat', 'LAMBDA_LM', 'OPTIMAL_TARIFF_LM');
writematrix(LAMBDA_LM', '05_output/lambda_lm.csv');
writematrix(OPTIMAL_TARIFF_LM', '05_output/optimal_tariff_lm.csv');

% ================================================================
% Evaluate and compare results
[comparison_table] = evaluate_comparison(TARIFFs, TARGETTARIFF, LAMBDA_OSSA, LAMBDA_LM, OPTIMAL_TARIFF_OSSA, OPTIMAL_TARIFF_LM);

save('05_output/comparison_results.mat', 'comparison_table');
writetable(comparison_table, '05_output/comparison_table.csv');

fprintf('Completed.\n');