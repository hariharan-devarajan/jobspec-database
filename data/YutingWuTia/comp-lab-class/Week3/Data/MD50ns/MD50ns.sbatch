#!/bin/bash

#SBATCH --job-name=run-gromacs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --time=72:00:00
#SBATCH --mem=20GB

cd /scratch/work/courses/CHEM-GA-2671-2022fa/yw5806/comp-lab-class/Week3/Data/MD50ns 

module purge

module load gromacs/openmpi/intel/2020.4

srun -n 10 gmx_mpi grompp -f md50ns.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_50.tpr

srun -n 10 gmx_mpi mdrun -v -deffnm md_0_50