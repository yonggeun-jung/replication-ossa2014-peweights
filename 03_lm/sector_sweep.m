function T = sector_sweep(j, sectors, lam0, TARIFFs, TARGETTARIFF, N, S, npts)

if nargin < 8 || isempty(npts), npts = 25; end

T_target = TARGETTARIFF(:, j);
UB       = max(T_target + 0.03, 2.25);
prohib   = (T_target > 2.25);
f = ~prohib;  nf = sum(f);
C = zeros(S, S);
C(f, f) = eye(nf) - ones(nf, 1) * ones(1, nf) / nf;

MFN = mean(T_target) * ones(S, 1);
MFN(prohib) = UB(prohib);

rows = {};
for s = sectors(:)'
    w0 = lam0(s);
    grid = exp(linspace(log(0.1), log(max(1.5 * w0, 20)), npts));
    for w = grid
        lam = lam0;
        lam(s) = 0;
        lam = lam * (S - w) / sum(lam);
        lam(s) = w;
        lam = max(lam, 1e-12);
        LM = ones(N, S);  LM(j, :) = lam';
        Tl = mymfnoptimaltariffj(j, TARIFFs, LM, 0, UB, MFN);
        rss = norm(C * (Tl - T_target))^2;
        rows(end+1, :) = {j, s, prohib(s), w, rss, Tl(s)}; %#ok<AGROW>
    end
end

T = cell2table(rows, 'VariableNames', ...
    {'Country', 'Sector', 'Prohibitive', 'Weight', 'RSS', 'Tariff'});
end
