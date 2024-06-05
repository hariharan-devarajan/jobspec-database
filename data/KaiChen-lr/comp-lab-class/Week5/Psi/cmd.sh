#!/bin/bash
#SBATCH --job-name="npt"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=40:00:00
##SBATCH --gres=gpu:1
module purge
module load gromacs/openmpi/intel/2018.3
mpirun -np 1 gmx_mpi mdrun -deffnm adp
