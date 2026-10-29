clear; clc;
global GECOUNT

RUN_ROBUST  = true;
RUN_IDENT   = true;
RUN_SOLVER  = true;
VERBOSE_DIAG = true;

SOLVER_NEVAL  = 8000;
SOLVER_MAXSEC = 4 * 3600;

eps_fd = 7e-4;
cnames = {'Brazil', 'China', 'EU', 'India', 'Japan', 'ROW', 'US'};

here = fileparts(mfilename('fullpath'));
cd(here);
addpath(genpath('01_data'));
addpath(genpath('02_ossa_original'));
addpath(genpath('03_lm'));

outdir = fullfile('05_output', 'appendix');
if ~exist(outdir, 'dir'), mkdir(outdir); end

DATA = load('01_data/DATA.mat');
TARIFF = DATA.TARIFF;
TARGETTARIFF = DATA.TARGETTARIFF;
N = size(TARIFF, 2);
S = size(TARIFF, 1) / N;
TEMP = reshape(TARIFF', [N, N, S]);
TARIFFs = permute(TEMP, [2, 1, 3]);

LAMBDA_O = read_matrix_either('lambda_ossa.csv', S, N);
LAMBDA_F = read_matrix_either('lambda_lm.csv', S, N);

if RUN_ROBUST
    fprintf('\n========== Stage 1: robust estimation ==========\n');
    LAMBDA_R = nan(S, N);
    TARIFF_R = nan(S, N);
    summary = nan(N, 9);
    TAR_O = read_matrix_either('optimal_tariff_ossa.csv', S, N);

    for j = 1:N
        warm = [LAMBDA_O(:, j), LAMBDA_F(:, j)];
        ropt1 = struct();
        [lam_r, tar_r, info] = estimate_lambda_lm_robust( ...
            j, TARIFFs, TARGETTARIFF, warm, 42000 + j, ropt1);
        LAMBDA_R(:, j) = lam_r;
        TARIFF_R(:, j) = tar_r;
        summary(j, :) = summary_row(j, info, lam_r, tar_r, LAMBDA_O, TAR_O, LAMBDA_F);
        if VERBOSE_DIAG && j ~= 5
            writetable(info.table, fullfile(outdir, sprintf('robust_diag_j%d.csv', j)));
        end
        fprintf('== %s pass 1 done: RSS(tau=1e-3)=%.4f ==\n\n', cnames{j}, info.rss_tau_selected);
    end

    lam_unc = LAMBDA_R(:, 5);
    tar_unc = TARIFF_R(:, 5);
    writetable(table((1:S)', lam_unc, tar_unc, 'VariableNames', ...
        {'sector', 'lambda_unconstrained', 'tariff_unconstrained'}), ...
        fullfile(outdir, 'robust_japan_unconstrained.csv'));

    j = 5;
    ropt = struct('prohib_lower', true, 'use_strategies', false, 'n_dirichlet', 1);
    warm = [LAMBDA_O(:, j), LAMBDA_F(:, j), lam_unc];
    [lam_r, tar_r, info] = estimate_lambda_lm_robust( ...
        j, TARIFFs, TARGETTARIFF, warm, 42105, ropt);
    LAMBDA_R(:, j) = lam_r;
    TARIFF_R(:, j) = tar_r;
    summary(j, :) = summary_row(j, info, lam_r, tar_r, LAMBDA_O, TAR_O, LAMBDA_F);
    writetable(info.table, fullfile(outdir, 'robust_diag_japan.csv'));

    writematrix(LAMBDA_R', fullfile(outdir, 'lambda_lm_robust.csv'));
    writematrix(TARIFF_R', fullfile(outdir, 'optimal_tariff_lm_robust.csv'));
    T = array2table(summary, 'VariableNames', {'Country', 'RSS_tau1e3', ...
        'RSS_exact', 'GradFree', 'GradExact', 'LamCorr_vs_Ossa', ...
        'TauCorr_vs_Ossa', 'LamCorr_vs_Fast', 'Seconds'});
    writetable(T, fullfile(outdir, 'robust_summary.csv'));
    disp(T);
else
    LAMBDA_R = read_matrix_at(fullfile(outdir, 'lambda_lm_robust.csv'), S, N);
end

if RUN_IDENT
    fprintf('\n========== Stage 2: identification diagnostics ==========\n');
    methods = {'Ossa', LAMBDA_O; 'LM_fast', LAMBDA_F; 'LM_robust', LAMBDA_R};
    tab = {};
    spectra = {};
    for j = 1:N
        T_target = TARGETTARIFF(:, j);
        UB = max(T_target + 0.03, 2.25);
        prohib = (T_target > 2.25);
        C = center_operator(prohib);
        MFN = mean(T_target) * ones(S, 1);
        MFN(prohib) = UB(prohib);
        for m = 1:size(methods, 1)
            t0 = tic;
            lam = methods{m, 2}(:, j);
            J = jcb_central(lam, j, TARIFFs, UB, MFN, N, S, eps_fd);
            Jc = C * J;
            sv = svd(Jc);
            r  = rank(Jc);
            n_small = sum(sv < 1e-6 * sv(1));
            sv_id = sv(sv >= 1e-6 * sv(1));
            kappa = sv_id(1) / sv_id(end);
            tab(end+1, :) = {j, cnames{j}, methods{m, 1}, sum(prohib), ...
                sum(~prohib), sum(prohib) + 1, r, sv(1), sv_id(end), ...
                n_small, kappa, toc(t0)};                        %#ok<AGROW>
            for k = 1:S
                spectra(end+1, :) = {j, cnames{j}, methods{m, 1}, k, sv(k)}; %#ok<AGROW>
            end
            fprintf('%-6s %-9s: |P|=%d, rank=%d, n_small=%d, cond_id=%.2e (%.0fs)\n', ...
                cnames{j}, methods{m, 1}, sum(prohib), r, n_small, kappa, toc(t0));
        end
    end
    T = cell2table(tab, 'VariableNames', {'Country', 'Name', 'Method', ...
        'P_count', 'F_count', 'StructNullDim', 'Rank_Jc', 'SV_max', ...
        'SV_min_identified', 'N_below_relTol', 'Cond_identified', 'Seconds'});
    writetable(T, fullfile(outdir, 'identification_table.csv'));
    Tsp = cell2table(spectra, 'VariableNames', {'Country', 'Name', 'Method', 'k', 'sigma_k'});
    writetable(Tsp, fullfile(outdir, 'identification_spectra.csv'));
    disp(T);

    fprintf('\n-- flatness walk (Japan, three estimates) --\n');
    jw = 5;
    Tw = flatness_walk(jw, {'Heuristic', LAMBDA_O(:, jw); ...
                            'Baseline',  LAMBDA_F(:, jw); ...
                            'Robust',    LAMBDA_R(:, jw)}, ...
                       TARIFFs, TARGETTARIFF, N, S, eps_fd);
    writetable(Tw, fullfile(outdir, 'flatness_walk.csv'));
    disp(Tw(Tw.Step == 0.8, {'Estimate','Direction','RSS_change','MaxDTauFree_pp'}));

    Tt = TARGETTARIFF(:, jw);
    pr = (Tt > 2.25);
    sfree = find(~pr & Tt == max(Tt(~pr)), 1);
    Ts = sector_sweep(jw, [find(pr, 1), sfree], LAMBDA_R(:, jw), ...
                      TARIFFs, TARGETTARIFF, N, S);
    writetable(Ts, fullfile(outdir, 'sector_sweep.csv'));
    fprintf('sector sweep written: sectors %d (prohibitive) and %d (free)\n', ...
            find(pr, 1), sfree);
end

if RUN_SOLVER
    fprintf('\n========== Stage 4: solver comparison ==========\n');
    srows = {};
    for j = [7, 5]
        T_target = TARGETTARIFF(:, j);
        geo = struct('j', j, 'S', S, 'N', N, ...
                     'UB', max(T_target + 0.03, 2.25), 'LB', T_target, ...
                     'prohib', T_target > 2.25, 'T_target', T_target, ...
                     'tau', 1e-3);
        geo.free = ~geo.prohib;
        geo.band_on = any(geo.prohib);
        geo.mtf = mean(T_target(geo.free));
        geo.MFN = mean(T_target) * ones(S, 1);
        if geo.band_on, geo.MFN(geo.prohib) = geo.UB(geo.prohib); end

        fprintf('---- %s (band = %s) ----\n', cnames{j}, string(geo.band_on));

        GECOUNT = 0;  t0 = tic;
        [lam1, it1] = solver_lm(geo, TARIFFs, eps_fd, SOLVER_NEVAL, SOLVER_MAXSEC);
        srows(end+1, :) = solver_row(cnames{j}, 'LM', lam1, it1, toc(t0), geo, TARIFFs, eps_fd); %#ok<AGROW>
        save_solver(srows, outdir);

        GECOUNT = 0;  t0 = tic;
        opts = optimoptions('lsqnonlin', 'Algorithm', 'trust-region-reflective', ...
            'SpecifyObjectiveGradient', false, 'FiniteDifferenceType', 'central', ...
            'FiniteDifferenceStepSize', eps_fd, 'UseParallel', true, ...
            'MaxFunctionEvaluations', SOLVER_NEVAL, 'MaxIterations', 400, ...
            'FunctionTolerance', 1e-12, 'StepTolerance', 1e-12, ...
            'OutputFcn', @(x, ov, st) toc(t0) > SOLVER_MAXSEC, 'Display', 'iter');
        [th2, ~, ~, ~, out2] = lsqnonlin(@(th) solver_resid(th, geo, TARIFFs), zeros(S, 1), [], [], opts);
        srows(end+1, :) = solver_row(cnames{j}, 'TRF', softmax_lam(th2, S), out2.iterations, toc(t0), geo, TARIFFs, eps_fd); %#ok<AGROW>
        save_solver(srows, outdir);

        GECOUNT = 0;  t0 = tic;
        nopts = optimset('MaxFunEvals', SOLVER_NEVAL, 'MaxIter', SOLVER_NEVAL, ...
            'TolFun', 1e-12, 'TolX', 1e-12, ...
            'OutputFcn', @(x, ov, st) toc(t0) > SOLVER_MAXSEC, 'Display', 'iter');
        [th3, ~, ~, out3] = fminsearch(@(th) sum(solver_resid(th, geo, TARIFFs).^2), zeros(S, 1), nopts);
        srows(end+1, :) = solver_row(cnames{j}, 'NM', softmax_lam(th3, S), out3.iterations, toc(t0), geo, TARIFFs, eps_fd); %#ok<AGROW>
        save_solver(srows, outdir);
    end
    movefile(fullfile(outdir, 'solver_comparison_partial.csv'), ...
             fullfile(outdir, 'solver_comparison.csv'));
    fprintf('Saved %s\n', fullfile(outdir, 'solver_comparison.csv'));
end

fprintf('\nrobust_run complete.\n');

function M = read_matrix_either(fname, S, N)
p1 = fullfile('05_output', 'main_text', fname);
p2 = fullfile('05_output', fname);
if isfile(p1), M = read_matrix_at(p1, S, N);
elseif isfile(p2), M = read_matrix_at(p2, S, N);
else, error('robust_run:missingInput', ...
        'Cannot find %s. Run main_run.m first.', fname);
end
end

function M = read_matrix_at(p, S, N)
M = readmatrix(p);
if isequal(size(M), [N, S]), M = M'; end
assert(isequal(size(M), [S, N]), 'Unexpected size in %s.', p);
end

function row = summary_row(j, info, lam_r, tar_r, LAMBDA_O, TAR_O, LAMBDA_F)
row = [j, info.rss_tau_selected, info.rss_exact_selected, ...
       info.grad_free_selected, info.grad_full_selected, ...
       corr(lam_r, LAMBDA_O(:, j)), corr(tar_r, TAR_O(:, j)), ...
       corr(lam_r, LAMBDA_F(:, j)), sum(info.table.seconds, 'omitnan')];
end

function C = center_operator(prohib)
S = numel(prohib);
if any(prohib)
    f = ~prohib;  nf = sum(f);
    C = zeros(S, S);
    C(f, f) = eye(nf) - ones(nf, 1) * ones(1, nf) / nf;
else
    C = eye(S) - ones(S, S) / S;
end
end

function J = jcb_central(lam, j, TARIFFs, UB, MFN, N, S, eps_fd)
J = zeros(S, S);
parfor k = 1:S
    dk = -eps_fd / (S - 1) * ones(S, 1);
    dk(k) = eps_fd;
    lp = max(lam + dk, 1e-12);  lp = S * lp / sum(lp);
    lm = max(lam - dk, 1e-12);  lm = S * lm / sum(lm);
    Tp = tariff_at(lp, j, TARIFFs, UB, MFN, N, S);
    Tm = tariff_at(lm, j, TARIFFs, UB, MFN, N, S);
    J(:, k) = (Tp - Tm) / (2 * eps_fd);
end
end

function T = tariff_at(lam, j, TARIFFs, UB, MFN, N, S)
LM = ones(N, S);
LM(j, :) = lam';
T = mymfnoptimaltariffj(j, TARIFFs, LM, 0, UB, MFN);
end

function y = softplus_safe(x, t)
y = max(x, 0) + t .* log1p(exp(-abs(x) ./ t));
end

function lam = softmax_lam(th, S)
th = th - max(th);
lam = S * exp(th) / sum(exp(th));
end

function T = counted_tariff(lam, geo, TARIFFs)
global GECOUNT 
GECOUNT = GECOUNT + 1;
T = tariff_at(lam, geo.j, TARIFFs, geo.UB, geo.MFN, geo.N, geo.S);
end

function r = solver_build_resid(T, geo)
S = geo.S;
r = zeros(S, 1);
mp = mean(T(geo.free));
r(geo.free) = T(geo.free) - (geo.T_target(geo.free) - geo.mtf + mp);
r(geo.prohib) = softplus_safe(T(geo.prohib) - geo.UB(geo.prohib), geo.tau);
if geo.band_on
    r(geo.prohib) = r(geo.prohib) + softplus_safe(geo.LB(geo.prohib) - T(geo.prohib), geo.tau);
end
end

function r = solver_resid(th, geo, TARIFFs)
T = counted_tariff(softmax_lam(th, geo.S), geo, TARIFFs);
r = solver_build_resid(T, geo);
end

function [lam, it] = solver_lm(geo, TARIFFs, eps_fd, max_neval, max_sec)
global GECOUNT 
S = geo.S;
tstart = tic;
lam = ones(S, 1);
gamma = 5e-4;
T0 = counted_tariff(lam, geo, TARIFFs);
r0 = solver_build_resid(T0, geo);
obj = r0' * r0;
C = center_operator(geo.prohib);
for it = 1:1000
    if GECOUNT + 2 * S + 2 > max_neval || toc(tstart) > max_sec, break; end
    J = solver_fd(lam, geo, TARIFFs, eps_fd);
    A = C * J;
    if any(geo.prohib)
        zu = (T0(geo.prohib) - geo.UB(geo.prohib)) / geo.tau;
        w  = 1 ./ (1 + exp(-zu));
        if geo.band_on
            zl = (geo.LB(geo.prohib) - T0(geo.prohib)) / geo.tau;
            w  = w - 1 ./ (1 + exp(-zl));
        end
        A(geo.prohib, :) = w .* J(geo.prohib, :);
    end
    grad = A' * r0;
    if norm(grad, inf) < 1e-4, break; end
    d = -(A' * A + gamma * eye(S)) \ grad;
    d = d - mean(d);
    d = d / max(1, norm(d, inf) / 0.05);
    step = 1.0;  accepted = false;
    for ls = 1:30
        if GECOUNT >= max_neval || toc(tstart) > max_sec, break; end
        lt = lam .* exp(step * d);  lt = S * lt / sum(lt);
        Tt = counted_tariff(lt, geo, TARIFFs);
        rt = solver_build_resid(Tt, geo);
        ot = rt' * rt;
        if ot <= obj * (1 - 1e-4 * step), accepted = true; break; end
        step = step / 2;
        if step < 2^-15, break; end
    end
    if accepted
        rel = (obj - ot) / max(1, obj);
        lam = lt; T0 = Tt; r0 = rt; obj = ot;
        gamma = max(1e-9, gamma / 10);
        if rel < 1e-7, break; end
    else
        gamma = min(1e2, 10 * gamma);
        if gamma >= 1e2, break; end
    end
end
end

function J = solver_fd(lam, geo, TARIFFs, eps_fd)
S = geo.S;
J = zeros(S, S);
parfor k = 1:S
    dk = -eps_fd / (S - 1) * ones(S, 1);
    dk(k) = eps_fd;
    lp = max(lam + dk, 1e-12);  lp = S * lp / sum(lp);
    lm = max(lam - dk, 1e-12);  lm = S * lm / sum(lm);
    Tp = counted_tariff(lp, geo, TARIFFs);
    Tm = counted_tariff(lm, geo, TARIFFs);
    J(:, k) = (Tp - Tm) / (2 * eps_fd);
end
end

function row = solver_row(cname, mname, lam, iters, secs, geo, TARIFFs, eps_fd)
global GECOUNT
% The evaluation solves below are not charged to the arm's count.
saved = GECOUNT;
T = counted_tariff(lam, geo, TARIFFs);
g1 = geo;  g1.band_on = false;
r1 = solver_build_resid(T, g1);
rss = r1' * r1;
J = solver_fd(lam, geo, TARIFFs, eps_fd);
C = center_operator(geo.prohib);
gfree = norm((C * J)' * r1, inf);
ib = true;
if any(geo.prohib)
    ib = all(T(geo.prohib) >= geo.LB(geo.prohib) - 0.02);
end
GECOUNT = saved;
fprintf('%s / %s: RSS(tau=1e-3)=%.4e, gradFree=%.3e, inband=%s, solves=%d, %.0fs\n', ...
    cname, mname, rss, gfree, string(ib), saved, secs);
row = {cname, mname, rss, gfree, ib, saved, iters, secs};
end

function save_solver(rows, outdir)
T = cell2table(rows, 'VariableNames', {'Country', 'Method', 'RSS_tau1e3', ...
    'GradFree', 'InBand', 'InnerSolves', 'Iterations', 'Seconds'});
writetable(T, fullfile(outdir, 'solver_comparison_partial.csv'));
end
