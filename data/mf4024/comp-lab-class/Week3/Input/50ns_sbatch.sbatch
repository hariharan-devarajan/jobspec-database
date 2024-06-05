#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute server
#SBATCH --ntasks-per-node=48             # runs 48 tasks on each server
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=md_50ns
#SBATCH --output=md_50ns.out
module purge
module load gromacs/openmpi/intel/2020.4
mpirun -np 1 gmx_mpi grompp -f md_50ns.mdp -c npt.gro -t npt.cpt -p topol.top -o md_50ns.tpr
mpirun -np 48 gmx_mpi mdrun -s -deffnm md_50ns