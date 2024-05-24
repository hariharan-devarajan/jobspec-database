#!/bin/bash
#SBATCH --job-name="rfm_IMB_Biband_0_5_job"
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --output=rfm_IMB_Biband_0_5_job.out
#SBATCH --error=rfm_IMB_Biband_0_5_job.err
#SBATCH --time=0:10:0
#SBATCH --exclusive
#SBATCH --partition=hpc
export SLURM_MPI_TYPE=pmix_v3
spack load intel-mpi-benchmarks
srun IMB-MPI1 biband -npmin 1
