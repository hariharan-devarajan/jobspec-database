#!/bin/bash
export NNODES=2
#SBATCH --job-name singularity-mpi
#SBATCH -N $NNODES # total number of nodes
#SBATCH --time=00:05:00 # Max execution time
#SBATCH --cpus-per-task=2
#SBATCH --output stdout.out
#SBATCH --error error.out

module load intel intelmpi
mpirun -n 2 singularity exec matrixMul.sif /opt/main
