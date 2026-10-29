clear; clc;

OUTDIR    = fullfile('05_output', 'sim');
N_TRIALS  = 50;
SCENARIOS = [1, 2];

here = fileparts(mfilename('fullpath'));
cd(fullfile(here, '..'));

kinds = {'trial', 'lambda', 'spectrum', 'starts', 'startlambda'};
for k = 1:numel(kinds)
    acc = [];  missing = 0;
    for sc = SCENARIOS
        for trial = 1:N_TRIALS
            f = fullfile(OUTDIR, sprintf('%s_s%d_t%03d.csv', kinds{k}, sc, trial));
            if ~exist(f, 'file'), missing = missing + 1; continue; end
            T = readtable(f);
            if isempty(acc), acc = T; else, acc = [acc; T]; end %#ok<AGROW>
        end
    end
    if isempty(acc)
        fprintf('%-12s : nothing found\n', kinds{k});
        continue;
    end
    out = fullfile(OUTDIR, sprintf('sim_%s_all.csv', kinds{k}));
    writetable(acc, out);
    fprintf('%-12s : %5d rows, %d files missing -> %s\n', ...
        kinds{k}, height(acc), missing, out);
end
