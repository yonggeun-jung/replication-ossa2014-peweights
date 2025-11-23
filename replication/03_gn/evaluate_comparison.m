function [comparison_table] = evaluate_comparison(TARIFFs, TARGET_ALL, LAMBDA_OSSA, LAMBDA_GN, TARIFF_OSSA, TARIFF_GN)
%==========================================================================
% Evaluate and compare Ossa vs GN results
%
% Inputs:
%   TARIFFs      - Tariff matrix (N x N x S)
%   TARGET_ALL   - Target tariffs (S x N)
%   LAMBDA_OSSA  - Ossa's lambda (S x N)
%   LAMBDA_GN    - GN lambda (S x N)
%   TARIFF_OSSA  - Ossa's optimal tariffs (S x N)
%   TARIFF_GN    - GN optimal tariffs (S x N)
%
% Outputs:
%   comparison_table - Table with RSS, GradNorm
%==========================================================================

N = size(TARIFFs, 1);
S = size(TARGET_ALL, 1);

% Initialize storage
RSS_Ossa = nan(N, 1);
RSS_GN = nan(N, 1);
GradNorm_Ossa = nan(N, 1);
GradNorm_GN = nan(N, 1);
Lambda_Corr = nan(N, 1);
Lambda_MAE = nan(N, 1);
Lambda_MaxDiff = nan(N, 1);
Tariff_Corr = nan(N, 1);
Tariff_MAE = nan(N, 1);
Tariff_MaxDiff = nan(N, 1);

% Evaluate each country
for j = 1:N
    T_target = TARGET_ALL(:, j);
    
    % Set parameters
    if j == 5
        params.eps_fd = 5e-4;
        params.use_smooth_hinge = true;
        MFN_guess = T_target;
    elseif ismember(j, [2, 3, 7])
        params.eps_fd = 7e-4;
        params.use_smooth_hinge = true;
        MFN_guess = mean(T_target) * ones(S, 1);
    else
        params.eps_fd = 5e-4;
        params.use_smooth_hinge = true;
        MFN_guess = mean(T_target) * ones(S, 1);
    end
    
    % Evaluate Ossa
    lambda_ossa = LAMBDA_OSSA(:, j);
    [RSS_Ossa(j), GradNorm_Ossa(j)] = evaluate_lambda(j, lambda_ossa, TARIFFs, TARGET_ALL, MFN_guess, params);
    
    % Evaluate GN
    lambda_gn = LAMBDA_GN(:, j);
    [RSS_GN(j), GradNorm_GN(j)] = evaluate_lambda(j, lambda_gn, TARIFFs, TARGET_ALL, MFN_guess, params);
    
    % Lambda comparison
    Lambda_Corr(j) = corr(lambda_ossa, lambda_gn);
    Lambda_MAE(j) = mean(abs(lambda_ossa - lambda_gn));
    Lambda_MaxDiff(j) = max(abs(lambda_ossa - lambda_gn));
    
    % Tariff comparison
    tariff_ossa = TARIFF_OSSA(:, j);
    tariff_gn = TARIFF_GN(:, j);
    Tariff_Corr(j) = corr(tariff_ossa, tariff_gn);
    Tariff_MAE(j) = mean(abs(tariff_ossa - tariff_gn));
    Tariff_MaxDiff(j) = max(abs(tariff_ossa - tariff_gn));
end

% Create comparison table
comparison_table = table((1:N)', ...
                         RSS_Ossa, RSS_GN, ...
                         GradNorm_Ossa, GradNorm_GN, ...
                         'VariableNames', {'Country', 'RSS_Ossa', 'RSS_GN', ...
                                          'GradNorm_Ossa', 'GradNorm_GN'});

end

% Helper function: Evaluate RSS and Gradient Norm for given lambda
function [rss, grad_norm] = evaluate_lambda(j, lambda, TARIFFs, TARGET_ALL, MFN_guess, params)

N = size(TARIFFs, 1);
S = size(TARGET_ALL, 1);
T_target = TARGET_ALL(:, j);
UB = max(T_target + 0.03, 2.25);

prohib_idx = (T_target > 2.25);
free_idx = ~prohib_idx;
mean_target_free = mean(T_target(free_idx));

% Centering matrix
if sum(prohib_idx) > 0
    n_free = sum(free_idx);
    C = zeros(S, S);
    C(free_idx, free_idx) = eye(n_free) - ones(n_free, 1) * ones(1, n_free) / n_free;
else
    C = eye(S) - (ones(S, 1) * ones(S, 1)') / S;
end

% Tau for smooth hinge (final value)
if params.use_smooth_hinge
    if j == 5
        tau = 1e-4;
    else
        tau = 1e-3;
    end
else
    tau = 0;
end

% Compute optimal tariff
LAMBDA = ones(N, S);
LAMBDA(j, :) = lambda';
T_opt = mymfnoptimaltariffj(j, TARIFFs, LAMBDA, 0, UB, MFN_guess);

% Compute residuals
r = compute_residuals(T_opt);

% RSS
rss = r' * r;

% Gradient norm via finite differences
eps_fd = params.eps_fd;
J = zeros(S, S);
for k = 1:S
    dk = -eps_fd / (S - 1) * ones(S, 1);
    dk(k) = eps_fd;
    lam_p = project_simplex(lambda + dk);
    lam_m = project_simplex(lambda - dk);
    
    LAMBDA_p = ones(N, S);
    LAMBDA_p(j, :) = lam_p';
    LAMBDA_m = ones(N, S);
    LAMBDA_m(j, :) = lam_m';
    
    Tp = mymfnoptimaltariffj(j, TARIFFs, LAMBDA_p, 0, UB, MFN_guess);
    Tm = mymfnoptimaltariffj(j, TARIFFs, LAMBDA_m, 0, UB, MFN_guess);
    
    J(:, k) = (Tp - Tm) / (2 * eps_fd);
end

Jc = C * J;
grad = Jc' * r;
grad_norm = norm(grad, inf);

    % Nested functions
    function r = compute_residuals(T)
        r = zeros(S, 1);
        mean_pred_free = mean(T(free_idx));
        should_be_free = T_target(free_idx) - mean_target_free + mean_pred_free;
        r(free_idx) = T(free_idx) - should_be_free;
        
        if tau > 0
            z = T(prohib_idx) - UB(prohib_idx);
            r(prohib_idx) = tau * log1p(exp(z / tau));
        else
            r(prohib_idx) = max(0, T(prohib_idx) - UB(prohib_idx));
        end
    end

    function x = project_simplex(x)
        x = max(x, 1e-12);
        x = S * x / sum(x);
    end

end