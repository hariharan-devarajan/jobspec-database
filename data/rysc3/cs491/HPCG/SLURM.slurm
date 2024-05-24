#!/bin/bash

#SBATCH -p condo
#SBATCH -N 1 
#SBATCH --ntasks-per-node 1
#SBATCH --gpus 1

# Load singularity module
module load singularity

# run singularity HPCG
srun --mpi=pmi2 singularity run --nv --bind .:/my-dat-files hpc-benchmarks:24.03.sif ./hpcg.sh --dat /my-dat-files/HPCG.dat

# Command to run interactively
# 
# allocation 
# salloc -p condo -N 2 --ntasks-per-node 2 --gpus 2 --mem 64GB
# 
# srun:
# srun --mpi=pmi2 /opt/spack/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/singularity-3.8.5-5x3v5slydkd4pay5yu7ilfrjc6hz2dh5/bin/singularity run --nv --bind .:/my-dat-files hpc-benchmarks:24.03.sif ./hpcg.sh --dat /my-dat-files/HPCG.dat 
#
