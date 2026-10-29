RUN_SIMULATION = true;

stages = {'main_run.m', 'robust_run.m'};
if RUN_SIMULATION
    stages{end+1} = 'simulation_run.m';
else
    fprintf('Stage 3 skipped (RUN_SIMULATION = false); submit slurm/03_sim_array.sbatch instead.\n');
end

% Each stage begins with clear, so the stage list is fixed before the loop.
for stage = stages
    fprintf('==============================================\n');
    fprintf(' MASTER RUN: %s, started %s\n', stage{1}, string(datetime('now')));
    fprintf('==============================================\n');
    run(stage{1});
end

fprintf('==============================================\n');
fprintf(' MASTER RUN complete: %s\n', string(datetime('now')));
fprintf('==============================================\n');
