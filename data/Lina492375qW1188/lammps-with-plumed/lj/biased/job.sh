#!/bin/bash
#SBATCH --job-name="lj"
#SBATCH -p RM-shared
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 8

module load openmpi/4.0.5-gcc10.2.0

mpirun -n 1 [PATH_TO_LAMMPS_FOLDER]/lammps*/src/lmp_mpi -in inscript.in

