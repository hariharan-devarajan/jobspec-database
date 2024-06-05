#!/bin/bash
#SBATCH -n 8 # 8 cores
#SBATCH --mem-per-cpu=16G # 16GB per core	# guest users can use maximum 48 cores and 128G memory in total
#SBATCH --time=48:00:00 # 48-hour run-time
#SBATCH --tmp=10G # Local scratch space per node
#SBATCH --job-name=runSpineOpt
#SBATCH --mail-type=BEGIN,END,FAIL # send an email when the job begins, ends, and fails
# #SBATCH --error=errors.log # append job's error messages to errors.log

# to execute this script: "sbatch < run_SpineOpt.sh"

module load gurobi gcc/11.4.0 julia/1.10.2

cd $SCRATCH/path/to/the/SpineOptProject

julia ./path/to/run_SpineOpt.jl
