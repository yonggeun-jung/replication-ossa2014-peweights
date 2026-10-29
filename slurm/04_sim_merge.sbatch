#!/bin/bash
#SBATCH --job-name=pew_04_merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=05_output/hpg_logs/04_merge.%j.out
#SBATCH --error=05_output/hpg_logs/04_merge.%j.err
module purge
module load matlab/2024b
matlab -nodisplay -nosplash -batch "addpath(genpath('04_simulations')); sim_merge"

LOG=05_output/hpg_logs/03_simulation_run.log
{
  echo "# Stage 3, the Monte Carlo experiment."
  echo "# Run as a SLURM job array, one task per trial, because a single serial"
  echo "# job would take months and exceed the partition limit. This file joins"
  echo "# the per-task logs 03_sim.<jobid>_<task>.out in task order, which is"
  echo "# scenario 1 trials 1-50 followed by scenario 2 trials 1-50."
  echo "# Assembled $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  for f in $(ls -1 05_output/hpg_logs/03_sim.*_*.out 2>/dev/null \
             | xargs grep -l '^S[12] trial' \
             | awk -F_ '{t = $NF; sub(/\.out$/, "", t); print t, $0}' \
             | sort -n | cut -d' ' -f2-); do
    echo "===== $f ====="
    cat "$f"
    echo
  done
} > "$LOG"
echo "wrote $LOG"
