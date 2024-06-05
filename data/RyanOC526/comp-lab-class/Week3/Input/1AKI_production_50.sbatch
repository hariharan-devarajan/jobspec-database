#!/bin/bash
#SBATCH --job-name=1AKI_50
#SBATCH --nodes=1
#SBATCH --tasks-per-node=48
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
##SBATCH --gres=gpu:1 ##
module purge
module load gromacs/openmpi/intel/2020.4
mpirun -np 10  gmx_mpi mdrun -deffnm md_50
