#!/bin/bash 
#SBATCH --job-name=run-gromacs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
module purge
module load gromacs/openmpi/intel/2020.4

mpirun -np 1 gmx_mpi mdrun -deffnm md_50