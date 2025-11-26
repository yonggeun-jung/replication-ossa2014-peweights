% ================================================================
% Replicating Political Economy Weights in Ossa (2014) without Upper Bound
% ================================================================
clear all; 
clc; 
close all;

rng(42);

addpath(genpath('../01_data'));
addpath(genpath('../02_ossa_original'));

% Load Data
DATA = load('../01_data/DATA.mat');
TARIFF = DATA.TARIFF;
TARGETTARIFF = DATA.TARGETTARIFF;

N = size(TARIFF, 2);
S = size(TARIFF, 1) / N;

TEMP = reshape(TARIFF', [N, N, S]);
TARIFFs = permute(TEMP, [2, 1, 3]);

% ================================================================
% Estimate lambda using the modified original Ossa method (mylambdaj_noUB.m)
tic;
LAMBDA_OSSA = zeros(S, N);
OPTIMAL_TARIFF_OSSA = zeros(S, N);

for j = 1:N
    TARGETTARIFFj = TARGETTARIFF(:, j);
    [LAMBDAj, MFNOPTIMALTARIFFj] = mylambdaj_noUB(j, TARGETTARIFFj);
    
    LAMBDA_OSSA(:, j) = LAMBDAj;
    OPTIMAL_TARIFF_OSSA(:, j) = MFNOPTIMALTARIFFj;
end
time_ossa = toc;
fprintf('Completed original method in %.1f seconds.\n', time_ossa);

save('../04_output/lambda_ossa_noUB.mat', 'LAMBDA_OSSA', 'OPTIMAL_TARIFF_OSSA');
writematrix(LAMBDA_OSSA', '../04_output/lambda_ossa_noUB.csv');
writematrix(OPTIMAL_TARIFF_OSSA', '../04_output/optimal_tariff_ossa_noUB.csv');

% ================================================================
% Estimate lambda using Gauss-Newton method (estimate_lambda_gn_noUB.m)
tic;
LAMBDA_GN = zeros(S, N);
OPTIMAL_TARIFF_GN = zeros(S, N);

for j = 1:N
    [lambda_j, tariff_j] = estimate_lambda_gn_noUB(j, TARIFFs, TARGETTARIFF, LAMBDA_OSSA);
    
    LAMBDA_GN(:, j) = lambda_j;
    OPTIMAL_TARIFF_GN(:, j) = tariff_j;
    
end
time_gn = toc;
fprintf('Completed GN method in %.1f seconds.\n', time_gn);

save('../04_output/lambda_gn_noUB.mat', 'LAMBDA_GN', 'OPTIMAL_TARIFF_GN');
writematrix(LAMBDA_GN', '../04_output/lambda_gn_noUB.csv');
writematrix(OPTIMAL_TARIFF_GN', '../04_output/optimal_tariff_gn_noUB.csv');

% ================================================================
% Evaluate and compare results
[comparison_table] = evaluate_comparison(TARIFFs, TARGETTARIFF, LAMBDA_OSSA, LAMBDA_GN, OPTIMAL_TARIFF_OSSA, OPTIMAL_TARIFF_GN);

save('../04_output/comparison_results_noUB.mat', 'comparison_table');
writetable(comparison_table, '../04_output/comparison_table_noUB.csv');

fprintf('Completed.\n');