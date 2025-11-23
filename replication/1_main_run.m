% ================================================================
% Main script: Replicating Political Economy Weights in Ossa (2014)
% ================================================================
clear all; 
clc; 
close all;

rng(42);

addpath(genpath('01_data'));
addpath(genpath('02_ossa_original'));
addpath(genpath('03_gn'));

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

save('04_output/lambda_ossa.mat', 'LAMBDA_OSSA', 'OPTIMAL_TARIFF_OSSA');
writematrix(LAMBDA_OSSA', '04_output/lambda_ossa.csv');
writematrix(OPTIMAL_TARIFF_OSSA', '04_output/optimal_tariff_ossa.csv');

% ================================================================
% Estimate lambda using Gauss-Newton method (estimate_lambda_gn.m)
tic;
LAMBDA_GN = zeros(S, N);
OPTIMAL_TARIFF_GN = zeros(S, N);

for j = 1:N
    [lambda_j, tariff_j] = estimate_lambda_gn(j, TARIFFs, TARGETTARIFF, LAMBDA_OSSA);
    
    LAMBDA_GN(:, j) = lambda_j;
    OPTIMAL_TARIFF_GN(:, j) = tariff_j;
    
end
time_gn = toc;
fprintf('Completed GN method in %.1f seconds.\n', time_gn);

save('04_output/lambda_gn.mat', 'LAMBDA_GN', 'OPTIMAL_TARIFF_GN');
writematrix(LAMBDA_GN', '04_output/lambda_gn.csv');
writematrix(OPTIMAL_TARIFF_GN', '04_output/optimal_tariff_gn.csv');

% ================================================================
% Evaluate and compare results
[comparison_table] = evaluate_comparison(TARIFFs, TARGETTARIFF, LAMBDA_OSSA, LAMBDA_GN, OPTIMAL_TARIFF_OSSA, OPTIMAL_TARIFF_GN);

save('04_output/comparison_results.mat', 'comparison_table');
writetable(comparison_table, '04_output/comparison_table.csv');

fprintf('Completed.\n');