#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem=5GB
#SBATCH --job-name=energy_minim
#SBATCH --

module purge
module load gromacs/openmpi/intel/2020.4

# move into directory
cd /home/tje3676/chem-class-2023/comp-lab-class/Week3/Data

# Run energy minim
gmx_mpi mdrun -deffnm nvt

