#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-bingalls
#SBATCH --mail-user=mrastwoo@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=16

module load  gcc/7.3.0 openmpi/3.1.2 gromacs/2020.2
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -n index.ndx -o md_0_10.tpr
gmx mdrun -np -deffnm md_0_10