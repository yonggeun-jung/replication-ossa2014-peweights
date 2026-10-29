# Cluster scripts

From the repository root, in order:

    sbatch slurm/01_main.sbatch
    sbatch slurm/02_robust.sbatch
    sbatch slurm/03_sim_array.sbatch
    sbatch slurm/04_sim_merge.sbatch

Stage 2 needs the output of stage 1. `03_sim_array.sbatch` runs 100 tasks, one trial each (scenario 1 trials 1–50, then scenario 2 trials 1–50), and gives the same numbers as `simulation_run.m`. Run `04_sim_merge.sbatch` after the array finishes.

Pass account and QoS on the command line:

    sbatch --account=<group> --qos=<group> slurm/<script>

Check the MATLAB module with `module spider matlab` and the partition limits with `sinfo -o "%P %l"`, and adjust `--time` in the scripts. The array uses 31 days because a Scenario 2 trial takes up to 25 days on one CPU.

Logs are written to `05_output/hpg_logs/`.
