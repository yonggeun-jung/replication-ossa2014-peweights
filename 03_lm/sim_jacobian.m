function J = sim_jacobian(lam, j, TARIFFs, UB, MFN, N, S, eps_fd)
J = zeros(S, S);
pewM = str2double(getenv('PEW_MAXWORKERS'));
if isnan(pewM), pewM = Inf; end
parfor (k = 1:S, pewM)
    dk = -eps_fd / (S - 1) * ones(S, 1);
    dk(k) = eps_fd;
    lp = max(lam + dk, 1e-12);  lp = S * lp / sum(lp);
    lm = max(lam - dk, 1e-12);  lm = S * lm / sum(lm);
    Tp = sim_tariff_at(lp, j, TARIFFs, UB, MFN, N, S);
    Tm = sim_tariff_at(lm, j, TARIFFs, UB, MFN, N, S);
    J(:, k) = (Tp - Tm) / (2 * eps_fd);
end
end
