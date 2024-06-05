#!/bin/bash

#SBATCH --job-name=run-gromacs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --time=72:00:00
#SBATCH --mem=20GB

cd /scratch/work/courses/CHEM-GA-2671-2022fa/yw5806/comp-lab-class/Week3/Data/step4-6 

module purge

module load gromacs/openmpi/intel/2020.4

mpirun -np 1 gmx_mpi grompp -f nvt.mdp -c energymin.gro -r energymin.gro -p topol.top -o nvt.tpr
mpirun gmx_mpi mdrun -v -deffnm nvt
mpirun -np 1 gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
mpirun gmx_mpi mdrun -v -deffnm npt


srun -n 10 gmx_mpi grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr

srun -n 10 gmx_mpi mdrun -v -deffnm md_0_1