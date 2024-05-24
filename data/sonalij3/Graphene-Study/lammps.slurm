#!/bin/bash
#SBATCH --job-name="8024_C-1100K"
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

echo
echo "Begin job ..."
date
echo

## ==== Run lammps ==== #
module load openmpi
mpirun -np 16 ../downloads/lammps-29Sep2021/src/lmp_mpi -in graphene_single_layer_ReaxFF_8024_C.input 

echo
echo "End of job ..."
date
echo
