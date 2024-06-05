#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --threads-per-core=1
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=0:15:0
#SBATCH --partition=nvgpu
#
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cray/22.11
module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/11.2.0 CUDAcore/11.8.0
#
ddt --connect \
srun --cpus-per-task=16  \
--cpu-bind=verbose,none ./cuda_visible_devices.sh \
./native.exe
