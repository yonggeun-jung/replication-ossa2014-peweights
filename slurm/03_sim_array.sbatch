#!/bin/bash
#SBATCH --job-name=pew_03_sim
#SBATCH --array=1-100%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=6gb
#SBATCH --time=31-00:00:00
#SBATCH --output=05_output/hpg_logs/03_sim.%A_%a.out
#SBATCH --error=05_output/hpg_logs/03_sim.%A_%a.err
module purge
module load matlab/2024b

export PEW_MAXWORKERS=0

N_TRIALS=50
OUTDIR=05_output/sim

matlab -nodisplay -nosplash -batch \
  "addpath(genpath('04_simulations')); sim_array_task(${SLURM_ARRAY_TASK_ID}, ${N_TRIALS}, '${OUTDIR}')"
