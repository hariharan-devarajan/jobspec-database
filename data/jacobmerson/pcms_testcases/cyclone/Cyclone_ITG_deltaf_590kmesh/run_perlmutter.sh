#!/bin/bash
#SBATCH -A m499
#SBATCH -C gpu
#SBATCH -t 00:05:00
#SBATCH -q regular
#SBATCH -n 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --job-name=cyclone-deltaf
#
#

module load cmake/3.24.3
module load cray-fftw

export SLURM_CPU_BIND="cores"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=16
export xgc_bin_path=/pscratch/sd/j/jmerson/xgc_delta_f/bin/xgc-es-cpp-gpu

# turn off cuda aware MPI for now
export PETSC_OPTIONS='-use_gpu_aware_mpi 0'
export MPICH_ABORT_ON_ERROR=1
ulimit -c unlimited

export n_mpi_ranks_per_node=4
export n_mpi_ranks=$((${SLURM_JOB_NUM_NODES} * ${n_mpi_ranks_per_node}))
echo 'Number of nodes: '                  ${SLURM_JOB_NUM_NODES}
echo 'MPI ranks (total): '                $n_mpi_ranks
echo 'MPI ranks per node: '               $n_mpi_ranks_per_node
echo 'Number of OMP threads: '            ${OMP_NUM_THREADS}
echo 'XGC executable: '                   ${xgc_bin_path}

srun hostname
scontrol show jobid ${SLURM_JOB_ID}
srun -N 2 -n 8 -c 32 --cpu-bind=cores --ntasks-per-node=4 --gpus-per-task=1 \
--gpu-bind=single:1 $xgc_bin_path >& Cyclone_590k_${SLURM_JOB_ID}.log
