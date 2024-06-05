#!/bin/bash
#SBATCH --job-name=run-gromacs-50ns
#SBATCH --nodes=1
#SBATCH --tasks-per-node=40
#SBACTH --mem=10GB
#SBATCH --time=24:00:00
##SBATCH --cpus-per-task=20

module purge 
module load gromacs/openmpi/intel/2020.4 
mpirun gmx_mpi mdrun -deffnm md_0_50
