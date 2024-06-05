#!/bin/bash
#SBATCH --job-name=openmpi_threads
#SBATCH --account=project_2001659
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1000
module load julia/1.8.5
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
srun julia --project=. test.jl
