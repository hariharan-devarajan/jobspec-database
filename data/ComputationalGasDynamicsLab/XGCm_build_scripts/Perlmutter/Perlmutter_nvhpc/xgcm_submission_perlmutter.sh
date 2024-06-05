#!/bin/bash
#SBATCH -A m499_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:30:00
#SBATCH -n 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3
#SBATCH --job-name=Cyclone_590k_nphi=8

module load cudatoolkit/11.5
module load cpe-cuda
module load craype-accel-nvidia80
export SLURM_CPU_BIND="cores"
export MPICH_ABORT_ON_ERROR=1
ulimit -c unlimited

srun ./XGCm --kokkos-threads=1 590kmesh.osh 590kmesh_6.cpn \
1 1 bfs bfs 1 1 0 3 input_20million_nrho=3 petsc petsc_xgcm.rc \
-use_gpu_aware_mpi 0
