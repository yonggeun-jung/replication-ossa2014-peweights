function [lambda_opt, tariff_opt] = estimate_lambda_gn(j, TARIFFs, TARGET_ALL, LAMBDA_OSSA)
%==========================================================================
% Estimate political economy weights using Gauss-Newton method
%
% Inputs:
%   j           - Country index
%   TARIFFs     - Tariff matrix (N x N x S)
%   TARGET_ALL  - Target tariffs (S x N)
%   LAMBDA_OSSA - Ossa's lambda estimates (S x N)
%
% Outputs:
%   lambda_opt  - Estimated weights (S x 1)
%   tariff_opt  - Optimal tariffs under lambda_opt (S x 1)
%==========================================================================
 
N = size(TARIFFs, 1);
S = size(TARGET_ALL, 1);
T_target = TARGET_ALL(:, j);

% Set parameters based on country
if j == 5                      % Japan
    params.max_iter = 1200;
    params.eps_fd = 5e-4;
    params.gamma0 = 1e-3;
    params.step_cap = 0.02;
    params.use_multistart = true;
    MFN_guess = T_target;
elseif ismember(j, [2, 3, 7])  % China, EU, US
    params.max_iter = 1000;
    params.eps_fd = 7e-4;
    params.gamma0 = 1e-3;
    params.step_cap = 0.06;
    params.use_multistart = true;
    MFN_guess = mean(T_target) * ones(S, 1);
else                           % Default
    params.max_iter = 800;
    params.eps_fd = 5e-4;
    params.gamma0 = 1e-3;
    params.step_cap = 0.05;
    params.use_multistart = true;
    MFN_guess = mean(T_target) * ones(S, 1);
end

% Multistart optimization (default is multistart)
if params.use_multistart
    [lambda_opt, obj, grad_norm, iter] = solve_multistart(j, TARIFFs, TARGET_ALL, MFN_guess, params, LAMBDA_OSSA);
else
    [lambda_opt, obj, grad_norm, iter, ~] = solve_gn(j, TARIFFs, TARGET_ALL, MFN_guess, params);
end

% Compute optimal tariff at final lambda
UB = max(T_target + 0.03, 2.25);
LAMBDA = ones(N, S);
LAMBDA(j, :) = lambda_opt';
tariff_opt = mymfnoptimaltariffj(j, TARIFFs, LAMBDA, 0, UB, MFN_guess);

end

% Multistart optimization
function [lambda_best, obj_best, grad_best, iter_best] = solve_multistart(j, TARIFFs, TARGET_ALL, MFN_guess, params, LAMBDA_OSSA)

S = size(TARGET_ALL, 1);
T_target = TARGET_ALL(:, j);
lam_ossa = LAMBDA_OSSA(:, j);

if j == 5
    num_starts = 7;
elseif ismember(j, [2, 3, 7]) 
    num_starts = 10;
else
    num_starts = 7;
end

lambda_best = ones(S, 1);
obj_best = inf;
grad_best = inf;
iter_best = 0;

for k = 1:num_starts
    % Initialize
    if k == 1
        lam_init = ones(S, 1) / S;
    elseif k == 2
        lam_init = T_target / sum(T_target);
    elseif k == 3
        lam_init = lam_ossa;
    elseif k == 4
        lam_init = lam_ossa .* (1 + 0.02 * randn(S, 1));
        lam_init = max(lam_init, 1e-6);
    elseif k == 5
        lam_init = lam_ossa .* (1 + 0.05 * randn(S, 1));
        lam_init = max(lam_init, 1e-6);
    elseif k == 6
        lam_init = (T_target + 1) ./ sum(T_target + 1);  % ← ./
    elseif k == 7
        if k > 1 && grad_best < inf
            lam_init = lambda_best .* (1 + 0.1 * randn(S, 1));
            lam_init = max(lam_init, 1e-6);
        else
            lam_init = rand(S, 1);
        end
    elseif k == 8
        weights = T_target.^0.5;
        lam_init = weights + 0.3*rand(S,1);
    elseif k == 9
        alpha_mix = 0.7;
        lam_init = alpha_mix * lam_ossa + (1-alpha_mix) * ones(S,1)/S;
    else  % k == 10
        lam_init = (1 ./ max(T_target, 0.1)) / sum(1 ./ max(T_target, 0.1));
    end
    
    lam_init = S * lam_init / sum(lam_init);
    
    % initialization
    [lam_try, obj_try, grad_try, iter_try, ~] = solve_gn(j, TARIFFs, TARGET_ALL, MFN_guess, params, lam_init);
    
    % Keep lowest gradient norm
    if grad_try < grad_best
        lambda_best = lam_try;
        obj_best = obj_try;
        grad_best = grad_try;
        iter_best = iter_try;
    end
end

end

% Gauss-Newton solver
function [lam, obj, grad_norm, it, exit_ok] = solve_gn(j, TARIFFs, TARGET_ALL, MFN_guess, params, lam_init)

if nargin < 6
    lam_init = ones(size(TARGET_ALL, 1), 1);
end

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

lam = S * lam_init / sum(lam_init);

% Parameters
max_iter = params.max_iter;
eps_fd = params.eps_fd;
gamma0 = params.gamma0;
step_cap = params.step_cap;

tol_rel = 1e-7;
tol_grad = 1e-4;
min_alpha = 2^-15;

% Tau schedule for smooth hinge
if j == 5
    tau_start = 1.0;
    tau_end = 1e-4;
else
    tau_start = 5e-2;
    tau_end = 1e-3;
end
tau = tau_start;

% Initial evaluation
T0 = compute_optimal_tariff(lam);
r0 = compute_residuals(T0, tau);
obj = r0' * r0;

exit_ok = 1;
grad = nan(S, 1);

for it = 1:max_iter
    % Update tau
    tau = tau_start * (tau_end / tau_start)^(it / max_iter);
    
    % Compute Jacobian via finite differences
    J = zeros(S, S);
    for k = 1:S
        dk = -eps_fd / (S - 1) * ones(S, 1);
        dk(k) = eps_fd;
        lam_p = project_simplex(lam + dk);
        lam_m = project_simplex(lam - dk);
        Tp = compute_optimal_tariff(lam_p);
        Tm = compute_optimal_tariff(lam_m);
        J(:, k) = (Tp - Tm) / (2 * eps_fd);
    end
    
    % Centered Jacobian and gradient
    Jc = C * J;
    grad = Jc' * r0;
    
    if norm(grad, inf) < tol_grad
        break;
    end
    
    % Gauss-Newton direction
    H = Jc' * Jc + gamma0 * eye(S);
    d = -H \ grad;
    d = C * d;
    d = d / max(1, norm(d, inf) / step_cap);
    
    % Line search
    alpha = 1.0;
    accepted = false;
    for ls = 1:30
        lam_try = lam .* exp(alpha * d);
        lam_try = S * lam_try / sum(lam_try);
        
        T_try = compute_optimal_tariff(lam_try);
        r_try = compute_residuals(T_try, tau);
        obj_try = r_try' * r_try;
        
        if obj_try <= obj * (1 - 1e-4 * alpha)
            accepted = true;
            break;
        end
        
        alpha = alpha / 2;
        if alpha < min_alpha
            break;
        end
    end
    
    if accepted
        rel_dec = (obj - obj_try) / max(1, obj);
        lam = lam_try;
        T0 = T_try;
        r0 = r_try;
        obj = obj_try;
        gamma0 = max(1e-9, gamma0 / 10);
        MFN_guess = T0;
        
        if rel_dec < tol_rel
            break;
        end
    else
        gamma0 = min(1e2, 10 * gamma0);
        if gamma0 >= 1e2
            exit_ok = 0;
            break;
        end
    end
end

grad_norm = norm(grad, inf);

    % Nested functions
    function T = compute_optimal_tariff(lam_)
        LM = ones(N, S);
        LM(j, :) = lam_';
        T = mymfnoptimaltariffj(j, TARIFFs, LM, 0, UB, MFN_guess);
    end

    function r = compute_residuals(T, tau_val)
        r = zeros(S, 1);
        mean_pred_free = mean(T(free_idx));
        should_be_free = T_target(free_idx) - mean_target_free + mean_pred_free;
        r(free_idx) = T(free_idx) - should_be_free;
        
        if tau_val > 0
            z = T(prohib_idx) - UB(prohib_idx);
            r(prohib_idx) = tau_val * log1p(exp(z / tau_val));
        else
            r(prohib_idx) = max(0, T(prohib_idx) - UB(prohib_idx));
        end
    end

    function x = project_simplex(x)
        x = max(x, 1e-12);
        x = S * x / sum(x);
    end

end