#!/bin/bash
#SBATCH --job-name=1AKI_energymin
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
##SBATCH --gres=gpu:1 ##
module purge
module load gromacs/openmpi/intel/2020.4
mpirun gmx_mpi mdrun -v -deffnm em
