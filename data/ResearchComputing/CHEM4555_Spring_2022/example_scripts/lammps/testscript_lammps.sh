#!/bin/bash

#SBATCH --job-name=lammps-test
#SBATCH --partition=shas
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:01:00
#SBATCH --output=lammps-test.%j.out

module purge
module load intel/17.4
module load impi/17.3
module load lammps/29Oct20

mpirun -np 2 lmp_mpi -in in.atm
