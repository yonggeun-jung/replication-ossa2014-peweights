clear all;
clc;
close all;

rng(42);

root = fileparts(mfilename('fullpath'));
cd(root);

addpath(genpath(fullfile(root, '01_data')));
addpath(genpath(fullfile(root, '02_ossa_original')));
addpath(genpath(fullfile(root, '03_lm')));

if ~exist(fullfile(root, '05_output'), 'dir')
    mkdir(fullfile(root, '05_output'));
end

DATA = load(fullfile(root, '01_data', 'DATA.mat'));
TARIFF = DATA.TARIFF;
TARGETTARIFF = DATA.TARGETTARIFF;

N = size(TARIFF, 2);
S = size(TARIFF, 1) / N;

TEMP = reshape(TARIFF', [N, N, S]);
TARIFFs = permute(TEMP, [2, 1, 3]);

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

save(fullfile(root, '05_output', 'lambda_ossa.mat'), 'LAMBDA_OSSA', 'OPTIMAL_TARIFF_OSSA');
writematrix(LAMBDA_OSSA', fullfile(root, '05_output', 'lambda_ossa.csv'));
writematrix(OPTIMAL_TARIFF_OSSA', fullfile(root, '05_output', 'optimal_tariff_ossa.csv'));

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

save(fullfile(root, '05_output', 'lambda_lm.mat'), 'LAMBDA_LM', 'OPTIMAL_TARIFF_LM');
writematrix(LAMBDA_LM', fullfile(root, '05_output', 'lambda_lm.csv'));
writematrix(OPTIMAL_TARIFF_LM', fullfile(root, '05_output', 'optimal_tariff_lm.csv'));

[comparison_table] = evaluate_comparison(TARIFFs, TARGETTARIFF, LAMBDA_OSSA, LAMBDA_LM, OPTIMAL_TARIFF_OSSA, OPTIMAL_TARIFF_LM);

save(fullfile(root, '05_output', 'comparison_results.mat'), 'comparison_table');
writetable(comparison_table, fullfile(root, '05_output', 'comparison_table.csv'));

fprintf('Completed.\n');
