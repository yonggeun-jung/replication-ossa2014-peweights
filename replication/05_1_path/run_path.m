%==========================================================================
% Find Optimization Path Landscape for Two Countries
%==========================================================================
clear; 
clc; 
close all;

addpath(genpath('../01_data'));
addpath(genpath('../02_ossa_original'));

% Load Data
DATA = load('../01_data/DATA.mat');
TARIFF_RAW = DATA.TARIFF;
TARGETTARIFF = DATA.TARGETTARIFF;

N = 7;
S = 33;

TEMP = reshape(TARIFF_RAW', [N, N, S]);
TARIFFs = permute(TEMP, [2, 1, 3]);

% Set up
countries = [5, 7];
country_names = {'Japan', 'US'};
n_grid = 30;

results = struct();

for c = 1:length(countries)
    j = countries(c);
    name = country_names{c};
    
    fprintf('\n=== %s (j=%d) ===\n', name, j);
    
    T_target = TARGETTARIFF(:, j);
    
    fprintf('Running Ossa...\n');
    [lambda_ossa, ~, hist_ossa] = mylambdaj_hist(j, T_target);
    
    fprintf('Running GN...\n');
    TARGET_ALL = zeros(S, N);
    TARGET_ALL(:, j) = T_target;
    LAMBDA_INIT = ones(S, N);
    [lambda_gn, ~, hist_gn] = estimate_lambda_gn_hist(j, TARIFFs, TARGET_ALL, LAMBDA_INIT);
    
    % Compute RSS
    rss_ossa = compute_rss(j, lambda_ossa, T_target, TARIFFs);
    rss_gn = compute_rss(j, lambda_gn, T_target, TARIFFs);
    
    fprintf('Ossa RSS: %.6f\n', rss_ossa);
    fprintf('GN RSS:   %.6f\n', rss_gn);
    
    % Select sectors
    diff = abs(lambda_gn - lambda_ossa);
    [~, idx] = sort(diff, 'descend');
    s1 = idx(1);
    s2 = idx(2);
    
    fprintf('Sectors: %d, %d (diff: %.4f, %.4f)\n', s1, s2, diff(s1), diff(s2));
    
    % Grid range
    all_s1 = [hist_ossa(:, s1)', hist_gn(:, s1)'];
    all_s2 = [hist_ossa(:, s2)', hist_gn(:, s2)'];
    
    margin = 0.5;
    range1 = linspace(max(0, min(all_s1) - margin), max(all_s1) + margin, n_grid);
    range2 = linspace(max(0, min(all_s2) - margin), max(all_s2) + margin, n_grid);
    
    [X, Y] = meshgrid(range1, range2);
    Z = zeros(n_grid, n_grid);
    
    lambda_fixed = lambda_gn;
    
    % Compute landscape
    fprintf('Computing %dx%d grid...\n', n_grid, n_grid);
    
    UB = max(T_target + 0.03, 2.25);
    prohib_idx = (T_target > 2.25);
    free_idx = ~prohib_idx;
    mean_target_free = mean(T_target(free_idx));
    
    tic;
    for k = 1:n_grid * n_grid
        [i, j_idx] = ind2sub([n_grid, n_grid], k);
        
        lam_test = lambda_fixed;
        lam_test(s1) = X(i, j_idx);
        lam_test(s2) = Y(i, j_idx);
        lam_test = S * lam_test / sum(lam_test);
        
        LAMBDA_TEST = ones(N, S);
        LAMBDA_TEST(j, :) = lam_test';
        tau_pred = mymfnoptimaltariffj(j, TARIFFs, LAMBDA_TEST, 0, UB, 0.1 * ones(S, 1));
        
        res = zeros(S, 1);
        mean_pred_free = mean(tau_pred(free_idx));
        should_be_free = T_target(free_idx) - mean_target_free + mean_pred_free;
        res(free_idx) = tau_pred(free_idx) - should_be_free;
        res(prohib_idx) = max(0, tau_pred(prohib_idx) - UB(prohib_idx));
        
        Z(k) = sum(res.^2);
    end
    Z = reshape(Z, [n_grid, n_grid]);
    
    fprintf('  Done: %.1f min\n', toc/60);
    
    % Store results
    results.(name).s1 = s1;
    results.(name).s2 = s2;
    results.(name).X = X;
    results.(name).Y = Y;
    results.(name).Z = Z;
    results.(name).hist_ossa = hist_ossa;
    results.(name).hist_gn = hist_gn;
    results.(name).lambda_ossa = lambda_ossa;
    results.(name).lambda_gn = lambda_gn;
    results.(name).rss_ossa = rss_ossa;
    results.(name).rss_gn = rss_gn;
end

% Save
save('path_data.mat', 'results');
fprintf('\n=== Saved: path_data.mat ===\n');

% Helper Function
function rss = compute_rss(j, lambda, T_target, TARIFFs)
    N = 7;
    S = length(lambda);
    
    LAMBDA = ones(N, S);
    LAMBDA(j, :) = lambda';
    
    UB = max(T_target + 0.03, 2.25);
    tau_pred = mymfnoptimaltariffj(j, TARIFFs, LAMBDA, 0, UB, T_target);
    
    prohib_idx = (T_target > 2.25);
    free_idx = ~prohib_idx;
    
    r = zeros(S, 1);
    mean_target_free = mean(T_target(free_idx));
    mean_pred_free = mean(tau_pred(free_idx));
    should_be_free = T_target(free_idx) - mean_target_free + mean_pred_free;
    r(free_idx) = tau_pred(free_idx) - should_be_free;
    r(prohib_idx) = max(0, tau_pred(prohib_idx) - UB(prohib_idx));
    
    rss = r' * r;
end