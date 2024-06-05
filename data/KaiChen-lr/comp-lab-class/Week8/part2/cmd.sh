#!/bin/bash
#SBATCH --job-name="melt"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=40:00:00 

source /scratch/work/courses/CHEM-GA-2671-2022fa/software/lammps-gcc-30Oct2022/setup_lammps.bash
for i in 0.95;
do
mpirun lmp -var density $i -in ../Inputs/2dWCA.in -log density_$i.log ;
done
