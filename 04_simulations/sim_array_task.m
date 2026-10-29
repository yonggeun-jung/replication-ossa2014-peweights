function sim_array_task(task_id, n_trials, outdir)

if ischar(task_id) || isstring(task_id), task_id = str2double(task_id); end
if nargin < 2 || isempty(n_trials), n_trials = 100; end
if ischar(n_trials) || isstring(n_trials), n_trials = str2double(n_trials); end
if nargin < 3 || isempty(outdir), outdir = fullfile('05_output', 'sim'); end

sc    = floor((task_id - 1) / n_trials) + 1;
trial = mod(task_id - 1, n_trials) + 1;

cfg = struct();
cfg.tc = 1;
sim_trial(sc, trial, outdir, cfg);
end
