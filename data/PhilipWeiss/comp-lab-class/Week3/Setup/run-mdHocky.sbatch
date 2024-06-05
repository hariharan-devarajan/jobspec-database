#!/bin/bash

#SBATCH --job-name=mdrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=10GB

module purge
module load gromacs/openmpi/intel/2020.4

gmx_mpi mdrun -deffnm md_0_1
