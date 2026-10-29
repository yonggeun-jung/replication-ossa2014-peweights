clear; clc;

N_TRIALS  = 50;
SCENARIOS = [1, 2];
OUTDIR    = fullfile('05_output', 'sim');

here = fileparts(mfilename('fullpath'));
cd(here);
addpath(genpath('04_simulations'));

cfg = struct('tc', 1);

for sc = SCENARIOS
    for trial = 1:N_TRIALS
        f = fullfile(OUTDIR, sprintf('trial_s%d_t%03d.csv', sc, trial));
        if exist(f, 'file')
            fprintf('S%d trial %3d: already present, skipping\n', sc, trial);
            continue;
        end
        sim_trial(sc, trial, OUTDIR, cfg);
    end
end

fprintf('\nall trials done, merging\n');
sim_merge;
