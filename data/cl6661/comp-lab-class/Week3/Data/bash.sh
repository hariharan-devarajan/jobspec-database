#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=run-gromace-50ns
#SBATCH --nodes=1
#SBATCH --tasks-per-node=48
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
module purge
module load gromacs/openmpi/intel/2020.4
mpirun gmx_mpi mdrun -deffnm nvt
mpirun -np 1 gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
mpirun gmx_mpi mdrun -deffnm npt
mpirun -np 1 gmx_mpi grompp -f md50ns.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_50.tpr
mpirun gmx_mpi mdrun -deffnm md_0_50