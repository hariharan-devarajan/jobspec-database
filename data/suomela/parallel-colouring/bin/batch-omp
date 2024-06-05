#!/bin/bash
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=2000
#SBATCH --array=10-31
#SBATCH --constraint=xeon
#SBATCH --output=tmp/slurm-%A_%a.out

export OMP_PROC_BIND=true
srun bin/run-test triton build-omp $SLURM_ARRAY_TASK_ID
