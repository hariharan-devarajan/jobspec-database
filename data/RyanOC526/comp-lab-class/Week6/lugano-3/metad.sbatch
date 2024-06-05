#!/bin/bash
#SBATCH --job-name=metad
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --mem=8GB
#SBATCH --time=04:00:00

source /scratch/work/courses/CHEM-GA-2671-2022fa/software/gromacs-2019.6-plumedSept2020/bin/GMXRC.bash.modules

gmx_mpi mdrun -s topol.tpr -nsteps 5000000 -plumed plumed.dat 
