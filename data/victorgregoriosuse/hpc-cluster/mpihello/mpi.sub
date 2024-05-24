#!/bin/bash
#SBATCH -J testing
#SBATCH -p normal
#SBATCH --time=5:00
#SBATCH -N 1
#SBATCH -n 10

source /usr/share/spack/setup-env.sh
spack load openmpi

[ -f mpihello ] && rm -f mpihello
mpicc -o mpihello mpihello.c

mpirun -np 10 ./mpihello
exit $?
