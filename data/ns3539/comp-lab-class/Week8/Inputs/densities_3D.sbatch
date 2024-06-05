#!/bin/bash 
#SBATCH --job-name=run-densities
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --mem=8GB
#SBATCH --time=2:00:00
module purge
source /scratch/work/courses/CHEM-GA-2671-2022fa/software/lammps-gcc-30Oct2022/setup_lammps.bash

for d in 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
do
    echo $d
    mpirun lmp -var density $d -in ../Inputs/3dWCA.in -log LOGFILE-3d$d.log
done
