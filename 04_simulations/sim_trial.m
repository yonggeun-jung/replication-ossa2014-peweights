function sim_trial(sc, trial, outdir, cfg)

if nargin < 4, cfg = struct(); end
if ~isfield(cfg, 'tc'),       cfg.tc = 1;          end
if ~isfield(cfg, 'eps_fd'),   cfg.eps_fd = 7e-4;   end
if ~isfield(cfg, 'ropt'),     cfg.ropt = struct(); end
% Equation (B1) is imposed here; the estimator's default leaves it off.
if ~isfield(cfg.ropt, 'prohib_lower'), cfg.ropt.prohib_lower = true; end

t_trial = tic;

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(genpath(fullfile(root, '01_data')));
addpath(genpath(fullfile(root, '02_ossa_original')));
addpath(genpath(fullfile(root, '03_lm')));
if ~exist(outdir, 'dir'), mkdir(outdir); end

DATA = load(fullfile(root, '01_data', 'DATA.mat'));
TARIFF = DATA.TARIFF;
TARGETTARIFF = DATA.TARGETTARIFF;
N = size(TARIFF, 2);
S = size(TARIFF, 1) / N;
TEMP = reshape(TARIFF', [N, N, S]);
TARIFFs = permute(TEMP, [2, 1, 3]);

tc = cfg.tc;
T_orig   = TARGETTARIFF(:, tc);
UB       = max(T_orig + 0.03, 2.25);
MFN_base = mean(T_orig) * ones(S, 1);
evalc('[lambda_base, ~] = mylambdaj(tc, T_orig);');

rng(10000 * sc + trial, 'twister');
switch sc
    case 1
        lam_true = lambda_base .* (1 + 0.05 * randn(S, 1));
        lam_true = max(lam_true, 1e-6);
    case 2
        lam_true = 0.5 + 9 * rand(S, 1);
    otherwise
        error('sim_trial:scenario', 'Unknown scenario %d.', sc);
end
lam_true = lam_true / mean(lam_true);

T_sim = sim_tariff_at(lam_true, tc, TARIFFs, UB, MFN_base, N, S);
TT = TARGETTARIFF;  TT(:, tc) = T_sim;

k_act  = sum(T_sim >= 2.25 - 1e-3);
prohib = (T_sim > 2.25);
P_ct   = sum(prohib);

MFN_sim = mean(T_sim) * ones(S, 1);
J  = sim_jacobian(lam_true, tc, TARIFFs, UB, MFN_sim, N, S, cfg.eps_fd);
C  = sim_center_operator(prohib);
[~, Sg, V] = svd(C * J);
sv  = diag(Sg);
idL = sv >= 1e-3 * sv(1);
idS = sv >= 1e-6 * sv(1);

ropt = cfg.ropt;

tic; evalc('[lam_h, ~] = mylambdaj(tc, T_sim);');            s_h = toc;
tic; [lam_f, ~] = estimate_lambda_lm(tc, TARIFFs, TT);       s_f = toc;
tic; [lam_r, ~, info] = estimate_lambda_lm_robust(tc, TARIFFs, TT, ...
         [lam_h, lam_f], 50000 + 1000 * sc + trial, ropt);   s_r = toc;

retained = info.selected_id;

St = info.table;
St = addvars(St, repmat(sc, height(St), 1), repmat(trial, height(St), 1), ...
    'Before', 1, 'NewVariableNames', {'Scenario', 'Trial'});
St = addvars(St, vecnorm(info.LAMBDA_all - lam_true)', ...
    'NewVariableNames', {'Err_vs_true'});
writetable(St, fullfile(outdir, sprintf('starts_s%d_t%03d.csv', sc, trial)));

SL = array2table(info.LAMBDA_all', 'VariableNames', ...
    arrayfun(@(k) sprintf('s%02d', k), 1:S, 'UniformOutput', false));
SL = addvars(SL, repmat(sc, height(SL), 1), repmat(trial, height(SL), 1), ...
    (1:height(SL))', 'Before', 1, ...
    'NewVariableNames', {'Scenario', 'Trial', 'Start_id'});
writetable(SL, fullfile(outdir, sprintf('startlambda_s%d_t%03d.csv', sc, trial)));

est = {'Heuristic', lam_h, s_h; 'Baseline', lam_f, s_f; 'Robust', lam_r, s_r};
rows = cell(3, 20);
for m = 1:3
    e = est{m, 2} - lam_true;
    rows(m, :) = {sc, trial, est{m, 1}, cfg.tc, ...
        var(lam_true), var(T_sim), k_act, P_ct, ...
        sum(~idL), sum(~idS), norm(e), ...
        norm(V(:, idL)' * e), norm(V(:, ~idL)' * e), ...
        norm(V(:, idS)' * e), norm(V(:, ~idS)' * e), ...
        norm(est{m, 2} - 1), ...
        norm(V(:, idL)' * (est{m, 2} - 1)), ...
        norm(V(:, ~idL)' * (est{m, 2} - 1)), ...
        retained, est{m, 3}};
end
T = cell2table(rows, 'VariableNames', {'Scenario', 'Trial', 'Method', 'Country', ...
    'Lambda_Var', 'Tariff_Var', 'K_active', 'P_count', ...
    'NullDim_loose', 'NullDim_strict', 'Norm_e', ...
    'Norm_e_id_loose', 'Norm_e_null_loose', ...
    'Norm_e_id_strict', 'Norm_e_null_strict', ...
    'Norm_lam_minus_1', 'Norm_d1_id_loose', 'Norm_d1_null_loose', ...
    'Retained_start', 'Seconds'});
writetable(T, fullfile(outdir, sprintf('trial_s%d_t%03d.csv', sc, trial)));

L = array2table([lam_true, lam_h, lam_f, lam_r]', 'VariableNames', ...
    arrayfun(@(k) sprintf('s%02d', k), 1:S, 'UniformOutput', false));
L = addvars(L, repmat(sc, 4, 1), repmat(trial, 4, 1), ...
    {'TRUE'; 'Heuristic'; 'Baseline'; 'Robust'}, ...
    'Before', 1, 'NewVariableNames', {'Scenario', 'Trial', 'Method'});
writetable(L, fullfile(outdir, sprintf('lambda_s%d_t%03d.csv', sc, trial)));

Sp = table(repmat(sc, S, 1), repmat(trial, S, 1), (1:S)', sv, ...
    'VariableNames', {'Scenario', 'Trial', 'k', 'sigma_k'});
writetable(Sp, fullfile(outdir, sprintf('spectrum_s%d_t%03d.csv', sc, trial)));

writematrix(C * J, fullfile(outdir, sprintf('resjac_s%d_t%03d.csv', sc, trial)));

fprintf(['S%d trial %3d | P=%d k_act=%d nd=%d | e: H=%.3f B=%.3f R=%.3f' ...
         ' | %.0fs\n'], sc, trial, P_ct, k_act, sum(~idL), ...
        norm(lam_h - lam_true), norm(lam_f - lam_true), ...
        norm(lam_r - lam_true), toc(t_trial));
end
