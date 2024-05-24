#!/bin/bash
#SBATCH --job-name=stream4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:05:00
#SBATCH --output=new_mpi_build/mpistream4_%j.out

module load compilers/armclang/24.04
module load libraries/openmpi/5.0.3/armclang-24.04

OMP_NUM_THREADS=16

# All nodes, all cores
mpirun -np 4 -map-by node ./STREAM/mpi_stream
