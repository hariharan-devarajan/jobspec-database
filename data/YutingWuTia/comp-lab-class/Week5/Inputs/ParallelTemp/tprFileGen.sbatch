#!/bin/bash

#SBATCH --job-name=run-gromacs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --time=72:00:00
#SBATCH --mem=20GB

cd /scratch/work/courses/CHEM-GA-2671-2022fa/yw5806/comp-lab-class/Inputs/ParallelTemp

module purge

module load gromacs/openmpi/intel/2018.3

srun -n 1 gmx_mpi grompp -f T300/adp_T300.mdp -c ../adp.gro -p ../adp.top -o T300/adp.tpr
srun -n 1 gmx_mpi grompp -f T350/adp_T350.mdp -c ../adp.gro -p ../adp.top -o T350/adp.tpr
srun -n 1 gmx_mpi grompp -f T400/adp_T400.mdp -c ../adp.gro -p ../adp.top -o T400/adp.tpr
srun -n 1 gmx_mpi grompp -f T450/adp_T450.mdp -c ../adp.gro -p ../adp.top -o T450/adp.tpr