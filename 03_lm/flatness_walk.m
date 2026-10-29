function T = flatness_walk(j, lam_by_method, TARIFFs, TARGETTARIFF, N, S, eps_fd, steps)

if nargin < 7 || isempty(eps_fd), eps_fd = 7e-4; end
if nargin < 8 || isempty(steps),  steps  = [0 0.05 0.10 0.20 0.40 0.80]; end

T_target = TARGETTARIFF(:, j);
UB       = max(T_target + 0.03, 2.25);
prohib   = (T_target > 2.25);
freeidx  = ~prohib;
C        = center_operator_local(prohib);

MFN = mean(T_target) * ones(S, 1);
MFN(prohib) = UB(prohib);

rows = {};
for e = 1:size(lam_by_method, 1)
    name = lam_by_method{e, 1};
    lam0 = lam_by_method{e, 2};

    J = jcb_local(lam0, j, TARIFFs, UB, MFN, N, S, eps_fd);
    [~, Sg, V] = svd(C * J);
    sv    = diag(Sg);
    nflat = sum(sv < 1e-6 * sv(1));

    T0   = tariff_local(lam0, j, TARIFFs, UB, MFN, N, S);
    rss0 = norm(C * (T0 - T_target))^2;

    dirs = {'flat', V(:, end); 'control', V(:, 1)};
    for d = 1:2
        v = dirs{d, 2};
        v = v - mean(v);
        v = v / norm(v);
        for t = steps
            lam = max(lam0 + t * v, 1e-12);
            lam = S * lam / sum(lam);
            Tl  = tariff_local(lam, j, TARIFFs, UB, MFN, N, S);
            rss = norm(C * (Tl - T_target))^2;
            rows(end+1, :) = {j, name, dirs{d, 1}, nflat, t, ...
                norm(lam - lam0), rss, rss - rss0, ...
                100 * max(abs(Tl(freeidx) - T0(freeidx))), ...
                100 * max(abs(Tl(prohib)  - T0(prohib)))}; %#ok<AGROW>
        end
    end
end

T = cell2table(rows, 'VariableNames', {'Country', 'Estimate', 'Direction', ...
    'FlatDim', 'Step', 'NormDLambda', 'RSS', 'RSS_change', ...
    'MaxDTauFree_pp', 'MaxDTauProhib_pp'});
end

function C = center_operator_local(prohib)
S = numel(prohib);
if any(prohib)
    f = ~prohib;  nf = sum(f);
    C = zeros(S, S);
    C(f, f) = eye(nf) - ones(nf, 1) * ones(1, nf) / nf;
else
    C = eye(S) - ones(S, S) / S;
end
end

function J = jcb_local(lam, j, TARIFFs, UB, MFN, N, S, h)
J = zeros(S, S);
parfor k = 1:S
    dk = -h / (S - 1) * ones(S, 1);  dk(k) = h;
    lp = max(lam + dk, 1e-12);  lp = S * lp / sum(lp);
    lm = max(lam - dk, 1e-12);  lm = S * lm / sum(lm);
    J(:, k) = (tariff_local(lp, j, TARIFFs, UB, MFN, N, S) ...
             - tariff_local(lm, j, TARIFFs, UB, MFN, N, S)) / (2 * h);
end
end

function T = tariff_local(lam, j, TARIFFs, UB, MFN, N, S)
LM = ones(N, S);  LM(j, :) = lam';
T = mymfnoptimaltariffj(j, TARIFFs, LM, 0, UB, MFN);
end
