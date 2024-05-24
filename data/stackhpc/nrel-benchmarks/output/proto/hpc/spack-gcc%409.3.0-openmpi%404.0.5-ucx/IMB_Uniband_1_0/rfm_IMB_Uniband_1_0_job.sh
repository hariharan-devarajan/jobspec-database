#!/bin/bash
#SBATCH --job-name="rfm_IMB_Uniband_1_0_job"
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --output=rfm_IMB_Uniband_1_0_job.out
#SBATCH --error=rfm_IMB_Uniband_1_0_job.err
#SBATCH --time=0:10:0
#SBATCH --exclusive
#SBATCH --partition=hpc
export SLURM_MPI_TYPE=pmix_v3
spack load intel-mpi-benchmarks
srun IMB-MPI1 uniband -npmin 1
