function [lambda_opt, tariff_opt, info] = estimate_lambda_lm_robust(j, TARIFFs, TARGET_ALL, lambda_warm, seed_base, opts)

if nargin < 4, lambda_warm = []; end
if nargin < 5 || isempty(seed_base), seed_base = 42000 + j; end
if nargin < 6, opts = struct(); end

def.tau_stages      = [5e-2, 1e-2, 1e-3, 1e-4];
def.max_iter_stage  = 120;
def.tol_grad_stage  = [1e-3, 1e-3, 1e-4, 1e-4];
def.eps_fd          = 7e-4;
def.gamma0          = 5e-4;
def.step_cap        = 0.05;
def.tol_rel         = 1e-7;
def.min_step        = 2^-15;
def.n_dirichlet     = 2;
def.grad_select_tol = Inf;   % Inf: candidates are retained on RSS alone
def.tau_report      = 1e-3;
def.prohib_lower    = false;
def.band_tol        = 0.02;
def.use_strategies  = true;
def.include_uniform = true;
def.include_perturb = true;
def.progress_file   = '';
fn = fieldnames(def);
for f = 1:numel(fn)
    if ~isfield(opts, fn{f}), opts.(fn{f}) = def.(fn{f}); end
end
if isscalar(opts.tol_grad_stage)
    opts.tol_grad_stage = repmat(opts.tol_grad_stage, 1, numel(opts.tau_stages));
end

N = size(TARIFFs, 1);
S = size(TARGET_ALL, 1);
T_target = TARGET_ALL(:, j);
UB = max(T_target + 0.03, 2.25);
LB = T_target;

prohib_idx = (T_target > 2.25);
free_idx   = ~prohib_idx;
mean_target_free = mean(T_target(free_idx));

if sum(prohib_idx) > 0
    n_free = sum(free_idx);
    C = zeros(S, S);
    C(free_idx, free_idx) = eye(n_free) - ones(n_free, 1) * ones(1, n_free) / n_free;
else
    C = eye(S) - (ones(S, 1) * ones(S, 1)') / S;
end

geom = struct('N', N, 'S', S, 'C', C, 'UB', UB, 'LB', LB, ...
              'band_on', opts.prohib_lower, ...
              'free_idx', free_idx, 'prohib_idx', prohib_idx, ...
              'T_target', T_target, 'mean_target_free', mean_target_free);

fprintf('\nCountry %d: robust LM estimation (seed %d, band %s)\n', ...
        j, seed_base, string(opts.prohib_lower));
fprintf('  |P| = %d prohibitive sectors, %d stages, tau: %s\n', ...
        sum(prohib_idx), numel(opts.tau_stages), mat2str(opts.tau_stages));

rng(seed_base, 'twister');

starts = {};
kinds  = {};

for k = 1:size(lambda_warm, 2)
    lam0 = max(lambda_warm(:, k), 1e-8);
    starts{end+1} = S * lam0 / sum(lam0);        
    kinds{end+1}  = sprintf('warm %d', k);       
end

if opts.include_uniform
    starts{end+1} = ones(S, 1);                  kinds{end+1} = 'uniform';
end

if opts.use_strategies
    strat_names = {'T_target direct', 'T_target^0.5', 'T_target^2', ...
                   'Inverse weights', 'Entropy-based', ...
                   'Random uniform', 'Random Dirichlet', ...
                   'T_target + noise 1', 'T_target + noise 2'};
    for k = 1:9
        switch k
            case 1
                lam0 = T_target / sum(T_target) * S;
            case 2
                w = sqrt(max(T_target, 1e-6));  lam0 = w / sum(w) * S;
            case 3
                w = T_target.^2;                lam0 = w / sum(w) * S;
            case 4
                w = 1 ./ max(T_target, 0.1);    lam0 = w / sum(w) * S;
            case 5
                p = T_target / sum(T_target);
                ew = -p .* log(p + 1e-10);      lam0 = S * ew / sum(ew);
            case 6
                lam0 = 0.5 + rand(S, 1);        lam0 = S * lam0 / sum(lam0);
            case 7
                lam0 = gamrnd(ones(S, 1), 1);   lam0 = S * lam0 / sum(lam0);
            case 8
                lam0 = T_target / sum(T_target) * S;
                lam0 = lam0 .* (1 + 0.1 * randn(S, 1));
                lam0 = max(lam0, 1e-6);         lam0 = S * lam0 / sum(lam0);
            case 9
                lam0 = T_target / sum(T_target) * S;
                lam0 = lam0 .* (1 + 0.3 * randn(S, 1));
                lam0 = max(lam0, 1e-6);         lam0 = S * lam0 / sum(lam0);
        end
        starts{end+1} = lam0;                    
        kinds{end+1}  = strat_names{k};          
    end
end

for k = 1:opts.n_dirichlet
    lam0 = gamrnd(ones(S, 1), 1);
    starts{end+1} = S * lam0 / sum(lam0);        
    kinds{end+1}  = sprintf('Dirichlet extra %d', k); 
end

n_starts = numel(starts) + double(opts.include_perturb);

do_progress = ~isempty(opts.progress_file);
if do_progress
    pf = fopen(opts.progress_file, 'a');
    if ftell(pf) == 0
        fprintf(pf, 'start_id,kind,rss_exact,rss_tau_report,grad_final,iters,seconds,exit_ok');
        fprintf(pf, ',lam_%d', 1:S);
        fprintf(pf, ',tar_%d', 1:S);
        fprintf(pf, '\n');
    end
end

rss_exact = nan(n_starts, 1);
rss_tau   = nan(n_starts, 1);
rss_band  = nan(n_starts, 1);
grad_gate = nan(n_starts, 1);
inband    = true(n_starts, 1);
iters     = zeros(n_starts, 1);
secs      = nan(n_starts, 1);
exit_ok   = ones(n_starts, 1);
LAM  = nan(S, n_starts);
TFIN = nan(S, n_starts);

best_rss   = inf;
lam_bestsf = ones(S, 1);

for i = 1:n_starts
    if i <= numel(starts)
        lam0 = starts{i};
        kind = kinds{i};
    else
        lam0 = lam_bestsf .* exp(0.2 * randn(S, 1));
        lam0 = max(lam0, 1e-6);
        lam0 = S * lam0 / sum(lam0);
        kind = 'best + perturb';
        kinds{end+1} = kind;                     
    end

    fprintf('  Start %2d/%2d (%s): ', i, n_starts, kind);
    t0 = tic;

    lam = lam0;
    MFN_guess = mean(T_target) * ones(S, 1);
    if geom.band_on
        MFN_guess(prohib_idx) = UB(prohib_idx);
    end
    ok_all = 1;
    it_tot = 0;
    T0 = [];

    p = struct('eps_fd', opts.eps_fd, 'gamma0', opts.gamma0, ...
               'step_cap', opts.step_cap, 'tol_rel', opts.tol_rel, ...
               'min_step', opts.min_step, 'max_iter', opts.max_iter_stage);

    for st = 1:numel(opts.tau_stages)
        p.tol_grad = opts.tol_grad_stage(st);
        [lam, ~, gnorm, it, ok, T0, MFN_guess] = ...
            solve_lm_stage(j, TARIFFs, lam, opts.tau_stages(st), p, geom, MFN_guess);
        it_tot = it_tot + it;
        ok_all = ok_all && ok;
    end

    secs(i) = toc(t0);

    r_ex = cpt_resid(T0, 0,               free_idx, prohib_idx, T_target, mean_target_free, UB, LB, false);
    r_tp = cpt_resid(T0, opts.tau_report, free_idx, prohib_idx, T_target, mean_target_free, UB, LB, false);
    rss_exact(i) = r_ex' * r_ex;
    rss_tau(i)   = r_tp' * r_tp;

    if geom.band_on
        r_bd = cpt_resid(T0, 0, free_idx, prohib_idx, T_target, mean_target_free, UB, LB, true);
        rss_band(i) = r_bd' * r_bd;
        inband(i)   = all(T0(prohib_idx) >= LB(prohib_idx) - opts.band_tol);
    end

    grad_gate(i) = gnorm;
    iters(i)     = it_tot;
    exit_ok(i)   = ok_all;
    LAM(:, i)    = lam;
    TFIN(:, i)   = T0;

    key_i = rss_exact(i);
    if geom.band_on, key_i = rss_band(i); end
    if key_i < best_rss
        best_rss   = key_i;
        lam_bestsf = lam;
    end

    if do_progress
        fprintf(pf, '%d,"%s",%.10e,%.10e,%.6e,%d,%.1f,%d', ...
                i, kind, rss_exact(i), rss_tau(i), grad_gate(i), ...
                it_tot, secs(i), ok_all);
        fprintf(pf, ',%.12g', lam);
        fprintf(pf, ',%.12g', T0);
        fprintf(pf, '\n');
    end

    fprintf('rss_exact=%.4e, rss(tau=%.0e)=%.4e, grad=%.2e, band=%s, it=%d, %.0fs\n', ...
            rss_exact(i), opts.tau_report, rss_tau(i), grad_gate(i), ...
            string(inband(i)), it_tot, secs(i));
end
if do_progress, fclose(pf); end

feas = grad_gate <= opts.grad_select_tol;
if geom.band_on
    feas = feas & inband;
end
if ~any(feas)
    warning('estimate_lambda_lm_robust:noFeasible', ...
        'Country %d: no start passed the gates; selecting global min RSS.', j);
    feas = true(n_starts, 1);
end
cand = find(feas);
[~, q] = sortrows([rss_exact(cand), grad_gate(cand)], [1, 2]);
pick = cand(q(1));

lambda_opt = LAM(:, pick);
tariff_opt = TFIN(:, pick);

MFN_rep = mean(T_target) * ones(S, 1);
Jrep = cpt_jcb_par(lambda_opt, j, TARIFFs, UB, MFN_rep, N, S, opts.eps_fd);
r_rep = cpt_resid(tariff_opt, opts.tau_report, free_idx, prohib_idx, ...
                  T_target, mean_target_free, UB, LB, false);
grad_free_sel = norm((C * Jrep)' * r_rep, inf);
A = C * Jrep;
if any(prohib_idx)
    zz  = tariff_opt(prohib_idx) - UB(prohib_idx);
    sig = 1 ./ (1 + exp(-zz / opts.tau_report));
    A(prohib_idx, :) = sig .* Jrep(prohib_idx, :);
end
grad_full_sel = norm(A' * r_rep, inf);

fprintf('  Selected start %d (%s): rss_exact=%.4e, rss(tau=%.0e)=%.4e, band=%s\n', ...
        pick, kinds{pick}, rss_exact(pick), opts.tau_report, rss_tau(pick), ...
        string(inband(pick)));
fprintf('  Gradients at selection: free-block=%.3e, exact=%.3e\n', ...
        grad_free_sel, grad_full_sel);

info = struct();
info.table = table((1:n_starts)', string(kinds(:)), rss_exact, rss_tau, ...
                   rss_band, inband, grad_gate, iters, secs, exit_ok, ...
                   'VariableNames', {'start_id', 'kind', 'rss_exact', ...
                   'rss_tau_report', 'rss_band', 'inband', 'grad_final', ...
                   'iters', 'seconds', 'exit_ok'});
info.selected_id        = pick;
info.selected_kind      = kinds{pick};
info.rss_exact_selected = rss_exact(pick);
info.rss_tau_selected   = rss_tau(pick);
info.rss_band_selected  = rss_band(pick);
info.inband_selected    = inband(pick);
info.grad_free_selected = grad_free_sel;
info.grad_full_selected = grad_full_sel;
info.LAMBDA_all         = LAM;
info.TARIFF_all         = TFIN;
info.seed_base          = seed_base;
info.opts               = opts;

fprintf('Country %d: robust estimation completed (%.0fs total).\n\n', j, sum(secs));

end

function [lam, obj, grad_norm, it, exit_ok, T0, MFN_guess] = ...
         solve_lm_stage(j, TARIFFs, lam_init, tau, p, geom, MFN_guess)

N = geom.N;  S = geom.S;
C = geom.C;  UB = geom.UB;  LB = geom.LB;
band_on  = geom.band_on;
free_idx = geom.free_idx;  prohib_idx = geom.prohib_idx;
T_target = geom.T_target;  mean_target_free = geom.mean_target_free;
npro = sum(prohib_idx);

lam = S * lam_init / sum(lam_init);
gamma = p.gamma0;

T0 = cpt_opt_tariff(lam, j, TARIFFs, UB, MFN_guess, N, S);
r0 = cpt_resid(T0, tau, free_idx, prohib_idx, T_target, mean_target_free, UB, LB, band_on);
obj = r0' * r0;

exit_ok = 1;
grad = nan(S, 1);
it = 0;

for it = 1:p.max_iter
    J = cpt_jcb_par(lam, j, TARIFFs, UB, MFN_guess, N, S, p.eps_fd);

    A = C * J;
    if npro > 0
        zu = (T0(prohib_idx) - UB(prohib_idx)) / tau;
        w  = 1 ./ (1 + exp(-zu));
        if band_on
            zl = (LB(prohib_idx) - T0(prohib_idx)) / tau;
            w  = w - 1 ./ (1 + exp(-zl));
        end
        A(prohib_idx, :) = w .* J(prohib_idx, :);
    end
    grad = A' * r0;

    if norm(grad, inf) < p.tol_grad
        break;
    end

    H = A' * A + gamma * eye(S);
    d = -H \ grad;
    d = d - mean(d);
    d = d / max(1, norm(d, inf) / p.step_cap);

    step = 1.0;
    accepted = false;
    for ls = 1:30
        lam_try = lam .* exp(step * d);
        lam_try = S * lam_try / sum(lam_try);

        T_try = cpt_opt_tariff(lam_try, j, TARIFFs, UB, MFN_guess, N, S);
        r_try = cpt_resid(T_try, tau, free_idx, prohib_idx, ...
                          T_target, mean_target_free, UB, LB, band_on);
        obj_try = r_try' * r_try;

        if obj_try <= obj * (1 - 1e-4 * step)
            accepted = true;
            break;
        end
        step = step / 2;
        if step < p.min_step
            break;
        end
    end

    if accepted
        rel_dec = (obj - obj_try) / max(1, obj);
        lam = lam_try;
        T0  = T_try;
        r0  = r_try;
        obj = obj_try;
        gamma = max(1e-9, gamma / 10);
        MFN_guess = T0;
        if band_on
            MFN_guess(prohib_idx) = UB(prohib_idx);
        end
        if rel_dec < p.tol_rel
            break;
        end
    else
        gamma = min(1e2, 10 * gamma);
        if gamma >= 1e2
            exit_ok = 0;
            break;
        end
    end
end

grad_norm = norm(grad, inf);

end

function J = cpt_jcb_par(lam, j, TARIFFs, UB, MFN_guess, N, S, eps_fd)

J = zeros(S, S);
pewM = str2double(getenv('PEW_MAXWORKERS'));
if isnan(pewM), pewM = Inf; end
parfor (k = 1:S, pewM)
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

function r = cpt_resid(T, tau_val, free_idx, prohib_idx, T_target, mean_target_free, UB, LB, band_on)

S = length(T);
r = zeros(S, 1);
mean_pred_free = mean(T(free_idx));
should_be_free = T_target(free_idx) - mean_target_free + mean_pred_free;
r(free_idx) = T(free_idx) - should_be_free;

if tau_val > 0
    % Overflow-safe form of tau*log(1 + exp(z/tau)), Equation (11).
    zu = T(prohib_idx) - UB(prohib_idx);
    r(prohib_idx) = max(zu, 0) + tau_val * log1p(exp(-abs(zu) / tau_val));
    if band_on
        zl = LB(prohib_idx) - T(prohib_idx);
        r(prohib_idx) = r(prohib_idx) + max(zl, 0) + tau_val * log1p(exp(-abs(zl) / tau_val));
    end
else
    r(prohib_idx) = max(0, T(prohib_idx) - UB(prohib_idx));
    if band_on
        r(prohib_idx) = r(prohib_idx) + max(0, LB(prohib_idx) - T(prohib_idx));
    end
end

end
