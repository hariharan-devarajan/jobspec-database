#!/bin/bash
#SBATCH --job-name=run_md
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

module purge
module load gromacs/openmpi/intel/2020.4
mpirun gmx_mpi mdrun -deffnm ../Setup/md_0_1
