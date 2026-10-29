function C = sim_center_operator(prohib)
S = numel(prohib);
if any(prohib)
    f = ~prohib;  nf = sum(f);
    C = zeros(S, S);
    C(f, f) = eye(nf) - ones(nf, 1) * ones(1, nf) / nf;
else
    C = eye(S) - ones(S, S) / S;
end
end
