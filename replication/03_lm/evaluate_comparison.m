function [comparison_table] = evaluate_comparison(TARIFFs, TARGET_ALL, LAMBDA_OSSA, LAMBDA_LM, TARIFF_OSSA, TARIFF_LM)

N = size(TARIFFs, 1);
S = size(TARGET_ALL, 1);

% Initialize storage
RSS_Ossa = nan(N, 1);
RSS_LM = nan(N, 1);
GradNorm_Ossa = nan(N, 1);
GradNorm_LM = nan(N, 1);
Lambda_Corr = nan(N, 1);
Tariff_Corr = nan(N, 1);
Tariff_Spearman = nan(N, 1);

fprintf('\nEvaluating Ossa and LM Results\n\n');

% Set parameters (same as LM estimation)
params.eps_fd = 7e-4;
params.use_smooth_hinge = true;
params.tau = 1e-3;

% Evaluate each country
for j = 1:N
    fprintf('Evaluating Country %d/%d...', j, N);
    
    T_target = TARGET_ALL(:, j);
    MFN_guess = mean(T_target) * ones(S, 1);
    
    % Evaluate Ossa
    lambda_ossa = LAMBDA_OSSA(:, j);
    [RSS_Ossa(j), GradNorm_Ossa(j)] = evaluate_lambda(j, lambda_ossa, TARIFFs, TARGET_ALL, MFN_guess, params);
    
    % Evaluate LM
    lambda_lm = LAMBDA_LM(:, j);
    [RSS_LM(j), GradNorm_LM(j)] = evaluate_lambda(j, lambda_lm, TARIFFs, TARGET_ALL, MFN_guess, params);
    
    % Lambda comparison (Pearson correlation)
    Lambda_Corr(j) = corr(lambda_ossa, lambda_lm);
    
    % Tariff comparison
    tariff_ossa = TARIFF_OSSA(:, j);
    tariff_lm = TARIFF_LM(:, j);
    
    % Correlation
    Tariff_Corr(j) = corr(tariff_ossa, tariff_lm);
    
    fprintf('RSS: Ossa=%.4e, LM=%.4e | Grad: Ossa=%.4e, LM=%.4e\n', ...
            RSS_Ossa(j), RSS_LM(j), GradNorm_Ossa(j), GradNorm_LM(j));
end

fprintf('\nEvaluation Completed.\n\n');

% Create comparison table
comparison_table = table((1:N)', ...
                         RSS_Ossa, RSS_LM, ...
                         GradNorm_Ossa, GradNorm_LM, ...
                         Lambda_Corr, ...
                         Tariff_Corr, Tariff_Spearman, ...
                         'VariableNames', {'Country', ...
                                          'RSS_Ossa', 'RSS_LM', ...
                                          'GradNorm_Ossa', 'GradNorm_LM', ...
                                          'Lambda_Corr', ...
                                          'Tariff_Corr'});

% Display summary
fprintf('RSS: Ossa mean=%.4e, LM mean=%.4e\n', mean(RSS_Ossa), mean(RSS_LM));
fprintf('GradNorm: Ossa mean=%.4e, LM mean=%.4e\n', mean(GradNorm_Ossa), mean(GradNorm_LM));
fprintf('Lambda Corr: mean=%.4f, min=%.4f, max=%.4f\n', ...
        mean(Lambda_Corr), min(Lambda_Corr), max(Lambda_Corr));
fprintf('Tariff Corr: mean=%.4f, min=%.4f, max=%.4f\n', ...
        mean(Tariff_Corr), min(Tariff_Corr), max(Tariff_Corr));

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

% Tau for smooth hinge (final value from LM)
tau = params.tau;

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