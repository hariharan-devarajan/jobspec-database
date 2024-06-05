#!/bin/bash

########################
#  SLURM JOB CONFIG
########################

#SBATCH --job-name=mpi_ec
#SBATCH --nodes=2
#SBATCH --ntasts-per-node=4
#SBATCH --time=00:05:00

module load gahpc

srun julia -p auto -L ./src/gahpc.jl
