function [lambda_opt, tariff_opt] = estimate_lambda_lm(j, TARIFFs, TARGET_ALL)

N = size(TARIFFs, 1);
S = size(TARGET_ALL, 1);
T_target = TARGET_ALL(:, j);

fprintf('\nCountry %d: Starting LM estimation\n', j);

% Base parameters
params.max_iter = 500;
params.eps_fd = 7e-4;
params.gamma0 = 5e-4;
params.step_cap = 0.05;
params.tol_grad = 1e-4;

% Initial guess: uniform weights
lam_init = ones(S, 1);
MFN_guess = mean(T_target) * ones(S, 1);

% First attempt with uniform initial guess
fprintf('Initial attempt with uniform weights...\n');
[lambda_opt, obj, grad_norm, iter, exit_ok] = solve_lm(j, TARIFFs, TARGET_ALL, MFN_guess, params, lam_init);
fprintf('Result: iter=%d, obj=%.4e, grad_norm=%.4e, exit_ok=%d\n', iter, obj, grad_norm, exit_ok);

% Adaptive multistart threshold based on initial result
multistart_threshold = 0.01;

if exit_ok == 0 || grad_norm > 2
    % Very bad initial result - use aggressive multistart
    fprintf('Poor initial result (exit_ok=%d, grad=%.4e), aggressive multistart...\n', exit_ok, grad_norm);
    num_multistart = 10;  % More attempts
    % Use more conservative parameters for ill-conditioned problems
    params.gamma0 = 5e-3;
    params.step_cap = 0.03;
elseif grad_norm > 0.1
    % Moderate result - standard multistart
    fprintf('Grad norm %.4e > 0.1, starting multistart...\n', grad_norm);
    num_multistart = 7;
elseif grad_norm > 0.01
    % Decent result - light multistart
    fprintf('Grad norm %.4e > 0.01, light multistart...\n', grad_norm);
    num_multistart = 5;
else
    % Good result - no multistart needed
    fprintf('Grad norm %.4e <= 0.01, convergence achieved.\n', grad_norm);
    num_multistart = 0;
end

if num_multistart > 0
    [lambda_opt, obj, grad_norm, iter] = solve_multistart(j, TARIFFs, TARGET_ALL, MFN_guess, params, lambda_opt, obj, grad_norm, iter, multistart_threshold, num_multistart);
    fprintf('Multistart completed: iter=%d, obj=%.4e, grad_norm=%.4e\n', iter, obj, grad_norm);
end

% Compute optimal tariff
UB = max(T_target + 0.03, 2.25);
LAMBDA = ones(N, S);
LAMBDA(j, :) = lambda_opt';
tariff_opt = mymfnoptimaltariffj(j, TARIFFs, LAMBDA, 0, UB, MFN_guess);

fprintf('Country %d: Estimation completed.\n\n', j);

end


function [lambda_best, obj_best, grad_best, iter_best] = solve_multistart(j, TARIFFs, TARGET_ALL, MFN_guess, params, lambda_init, obj_init, grad_init, iter_init, threshold, num_starts)

S = size(TARGET_ALL, 1);
T_target = TARGET_ALL(:, j);

% Start with initial attempt result
lambda_best = lambda_init;
obj_best = obj_init;
grad_best = grad_init;
iter_best = iter_init;

fprintf('Initial result: obj=%.4e, grad=%.4e, iter=%d [BASELINE]\n', obj_init, grad_init, iter_init);

% More diverse initialization strategies
start_names = {'T_target direct', 'T_target^0.5', 'T_target^2', ...
               'Inverse weights', 'Entropy-based', ...
               'Random uniform', 'Random Dirichlet', ...
               'T_target + noise 1', 'T_target + noise 2', 'Best + perturb'};

for k = 1:num_starts
    fprintf('Multistart %d/%d (%s): ', k, num_starts, start_names{k});
    
    if k == 1
        % Direct T_target proportional
        lam_init = T_target / sum(T_target) * S;
    elseif k == 2
        % Square root of T_target (less extreme)
        weights = sqrt(max(T_target, 1e-6));
        lam_init = weights / sum(weights) * S;
    elseif k == 3
        % Squared T_target (more extreme)
        weights = T_target.^2;
        lam_init = weights / sum(weights) * S;
    elseif k == 4
        % Inverse weights (opposite of T_target)
        weights = 1 ./ max(T_target, 0.1);
        lam_init = weights / sum(weights) * S;
    elseif k == 5
        % Entropy-maximizing perturbation
        p = T_target / sum(T_target);
        entropy_weight = -p .* log(p + 1e-10);
        weights = entropy_weight / sum(entropy_weight);
        lam_init = S * weights;
    elseif k == 6
        % Random uniform [0.5, 1.5]
        lam_init = (0.5 + rand(S, 1));
        lam_init = S * lam_init / sum(lam_init);
    elseif k == 7
        % Random Dirichlet (generates diverse simplex points)
        lam_init = gamrnd(ones(S, 1), 1);
        lam_init = S * lam_init / sum(lam_init);
    elseif k == 8
        % T_target with small noise
        lam_init = T_target / sum(T_target) * S;
        lam_init = lam_init .* (1 + 0.1 * randn(S, 1));
        lam_init = max(lam_init, 1e-6);
        lam_init = S * lam_init / sum(lam_init);
    elseif k == 9
        % T_target with large noise
        lam_init = T_target / sum(T_target) * S;
        lam_init = lam_init .* (1 + 0.3 * randn(S, 1));
        lam_init = max(lam_init, 1e-6);
        lam_init = S * lam_init / sum(lam_init);
    else
        % Perturbation around current best
        lam_init = lambda_best .* exp(0.2 * randn(S, 1));
        lam_init = max(lam_init, 1e-6);
        lam_init = S * lam_init / sum(lam_init);
    end
        
    [lam_try, obj_try, grad_try, iter_try, ~] = solve_lm(j, TARIFFs, TARGET_ALL, MFN_guess, params, lam_init);
    
    fprintf('obj=%.4e, grad=%.4e, iter=%d', obj_try, grad_try, iter_try);
    
    if grad_try < grad_best
        lambda_best = lam_try;
        obj_best = obj_try;
        grad_best = grad_try;
        iter_best = iter_try;
        fprintf('[BEST]');
        
        if grad_best <= threshold
            fprintf('\nGrad norm %.4e <= %.4e, early stop.\n', grad_best, threshold);
            break;
        end
    end
    fprintf('\n');
end

end

% Main solver
function [lam, obj, grad_norm, it, exit_ok] = solve_lm(j, TARIFFs, TARGET_ALL, MFN_guess, params, lam_init)

N = size(TARIFFs, 1);
S = size(TARGET_ALL, 1);
T_target = TARGET_ALL(:, j);
UB = max(T_target + 0.03, 2.25);

prohib_idx = (T_target > 2.25);
free_idx = ~prohib_idx;
mean_target_free = mean(T_target(free_idx));

if sum(prohib_idx) > 0
    n_free = sum(free_idx);
    C = zeros(S, S);
    C(free_idx, free_idx) = eye(n_free) - ones(n_free, 1) * ones(1, n_free) / n_free;
else
    C = eye(S) - (ones(S, 1) * ones(S, 1)') / S;
end

lam = S * lam_init / sum(lam_init);

max_iter = params.max_iter;
eps_fd = params.eps_fd;
gamma0 = params.gamma0;
step_cap = params.step_cap;
tol_grad = params.tol_grad;

tol_rel = 1e-7;
min_alpha = 2^-15;

tau_start = 5e-2;
tau_end = 1e-3;
tau = tau_start;

broyden_refresh = 25;
J = [];
lam_prev = lam;
T_prev = [];

T0 = cpt_opt_tariff(lam, j, TARIFFs, UB, MFN_guess, N, S);
r0 = cpt_resid(T0, tau, free_idx, prohib_idx, T_target, mean_target_free, UB);
obj = r0' * r0;

exit_ok = 1;
grad = nan(S, 1);

for it = 1:max_iter
    tau = tau_start * (tau_end / tau_start)^(it / max_iter);
    
    if isempty(J) || mod(it, broyden_refresh) == 0
        J = cpt_jcb_par(lam, j, TARIFFs, UB, MFN_guess, N, S, eps_fd);
    else
        s = lam - lam_prev;
        y = T0 - T_prev;
        denom = s' * s;
        if denom > 1e-14
            J = J + ((y - J * s) * s') / denom;
        end
    end
    
    lam_prev = lam;
    T_prev = T0;
    
    Jc = C * J;
    grad = Jc' * r0;
    
    if norm(grad, inf) < tol_grad
        break;
    end
    
    H = Jc' * Jc + gamma0 * eye(S);
    d = -H \ grad;
    d = C * d;
    d = d / max(1, norm(d, inf) / step_cap);
    
    alpha = 1.0;
    accepted = false;
    for ls = 1:30
        lam_try = lam .* exp(alpha * d);
        lam_try = S * lam_try / sum(lam_try);
        
        T_try = cpt_opt_tariff(lam_try, j, TARIFFs, UB, MFN_guess, N, S);
        r_try = cpt_resid(T_try, tau, free_idx, prohib_idx, T_target, mean_target_free, UB);
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

end

% Compute Jacobian with parallel
function J = cpt_jcb_par(lam, j, TARIFFs, UB, MFN_guess, N, S, eps_fd)

J = zeros(S, S);
parfor k = 1:S    % if parallel package is unavailable, you may switch to for loop
    dk = -eps_fd / (S - 1) * ones(S, 1);
    dk(k) = eps_fd;
    
    lam_p = lam + dk;
    lam_p = max(lam_p, 1e-12);
    lam_p = S * lam_p / sum(lam_p);
    
    lam_m = lam - dk;
    lam_m = max(lam_m, 1e-12);
    lam_m = S * lam_m / sum(lam_m);
    
    Tp = cpt_opt_tariff(lam_p, j, TARIFFs, UB, MFN_guess, N, S);
    Tm = cpt_opt_tariff(lam_m, j, TARIFFs, UB, MFN_guess, N, S);
    J(:, k) = (Tp - Tm) / (2 * eps_fd);
end

end


function T = cpt_opt_tariff(lam_, j, TARIFFs, UB, MFN_guess, N, S)

LM = ones(N, S);
LM(j, :) = lam_';
T = mymfnoptimaltariffj(j, TARIFFs, LM, 0, UB, MFN_guess);

end

% Compute residual
function r = cpt_resid(T, tau_val, free_idx, prohib_idx, T_target, mean_target_free, UB)

S = length(T);
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
