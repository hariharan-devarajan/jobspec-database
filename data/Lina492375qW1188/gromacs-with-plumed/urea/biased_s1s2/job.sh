#!/bin/bash
#SBATCH --job-name="urea"
#SBATCH -p RM-shared
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 18

module load plumed
module load gcc
module load gromacs/2020.2-cpu

sh GenTprFile.sh
mpirun -np $SLURM_NPROCS -c 3 gmx_mpi mdrun -ntomp 6 -v -deffnm urea -plumed plumed.dat

