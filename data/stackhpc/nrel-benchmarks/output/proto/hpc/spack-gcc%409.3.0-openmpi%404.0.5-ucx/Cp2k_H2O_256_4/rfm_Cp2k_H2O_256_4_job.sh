#!/bin/bash
#SBATCH --job-name="rfm_Cp2k_H2O_256_4_job"
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --output=rfm_Cp2k_H2O_256_4_job.out
#SBATCH --error=rfm_Cp2k_H2O_256_4_job.err
#SBATCH --exclusive
#SBATCH --partition=hpc
export SLURM_MPI_TYPE=pmix_v3
spack load cp2k
time \
srun cp2k.popt H2O-256.inp
