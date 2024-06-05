#!/bin/bash

#SBATCH --job-name=run-gromacs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=48
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
##SBATCH --gres=gpu:1

module purge
module load gromacs/openmpi/intel/2020.4
##mpirun -np 1 gmx_mpi grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
##mpirun gmx_mpi mdrun -deffnm nvt
##echo "NVT Equilibriation Done" >> output.out
##mpirun -np 1 gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
##mpirun gmx_mpi mdrun -deffnm npt
##echo "NPT Equilibriation Done" >> output.out
mpirun -np 1 gmx_mpi grompp -f md-50ns.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_50.tpr
mpirun gmx_mpi mdrun -deffnm md_0_1
echo "Production Done" >> output.out
