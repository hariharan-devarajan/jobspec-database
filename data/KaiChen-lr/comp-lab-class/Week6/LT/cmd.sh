#!/bin/bash
#SBATCH --job-name="E1A"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=40:00:00
##SBATCH --gres=gpu:1

source /scratch/work/courses/CHEM-GA-2671-2022fa/software/gromacs-2019.6-plumedSept2020/bin/GMXRC.bash.modules
gmx_mpi mdrun -s topolA.tpr -nsteps 5000000 -plumed plumed.dat
