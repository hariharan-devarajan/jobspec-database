#!/bin/bash
#SBATCH --job-name=run-density
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --mem=8GB
#SBATCH --time=04:00:00 
source /scratch/work/courses/CHEM-GA-2671-2022fa/software/lammps-gcc-30Oct2022/setup_lammps.bash
mpirun lmp -var density 0.5 -in 3dWCA.in