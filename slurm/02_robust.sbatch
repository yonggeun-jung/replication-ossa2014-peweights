#!/bin/bash
#SBATCH --job-name=pew_02_robust
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=96:00:00
#SBATCH --output=05_output/hpg_logs/02_robust.%j.out
#SBATCH --error=05_output/hpg_logs/02_robust.%j.err
module purge
module load matlab/2024b
matlab -nodisplay -nosplash -batch "run('robust_run.m')"
