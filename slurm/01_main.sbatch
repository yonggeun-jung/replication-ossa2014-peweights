#!/bin/bash
#SBATCH --job-name=pew_01_main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --output=05_output/hpg_logs/01_main.%j.out
#SBATCH --error=05_output/hpg_logs/01_main.%j.err
module purge
module load matlab/2024b
matlab -nodisplay -nosplash -batch "run('main_run.m')"
