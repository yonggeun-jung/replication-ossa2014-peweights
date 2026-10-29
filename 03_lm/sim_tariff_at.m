function T = sim_tariff_at(lam, j, TARIFFs, UB, MFN, N, S)
LM = ones(N, S);
LM(j, :) = lam';
T = mymfnoptimaltariffj(j, TARIFFs, LM, 0, UB, MFN);
end
