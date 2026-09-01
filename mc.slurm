#!/bin/bash

#SBATCH --job-name mc
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --array=0-5
#SBATCH --mem 64gb
#SBATCH --time 72:00:00

TEMPERATURES=(100 200 300 400 500 600)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun venv/bin/python mc.py ${TEMPERATURES[$SLURM_ARRAY_TASK_ID]}
