#!/bin/bash -l
#SBATCH --job-name="cclm-debug"
#SBATCH --account=s824
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gianluca.mussetti@env.ethz.ch
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=debug.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

srun -u ./cclm_debug

