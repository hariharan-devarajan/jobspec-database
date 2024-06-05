#!/bin/bash

#SBATCH --job-name=mdrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=16GB

module purge
module load gromacs/openmpi/intel/2020.4

-np 1 gmx_mpi grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_50.tpr


gmx_mpi mdrun -deffnm md_0_50
