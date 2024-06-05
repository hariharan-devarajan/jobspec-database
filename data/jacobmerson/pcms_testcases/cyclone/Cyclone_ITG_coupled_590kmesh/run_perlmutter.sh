#!/bin/bash
#SBATCH -A m499
#SBATCH -C gpu
#SBATCH -t 04:00:00
#SBATCH -q regular
#SBATCH -N 21
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --job-name=coupled
#
#
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=perlmutter@jacobmerson.com

module load cmake/3.24.3
module load cray-fftw

export SLURM_CPU_BIND="cores"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=16

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

srun hostname
scontrol show jobid ${SLURM_JOB_ID}


removeBPSST() {
rm -r *.sst
rm -r core_c2s.bp
rm -r core_s2c.bp
rm -r edge_c2s.bp 
rm -r edge_s2c.bp
}
run_coupler() {
export coupler_bin_path=/pscratch/sd/j/jmerson/coupler-build/pcms/test/xgc_n0_server
echo 'Coupler executable: '                   ${coupler_bin_path}
set -x
OMP_NUM_THREADS=1
srun -N 1 -c 1 -n 4 --cpu-bind=cores --ntasks-per-node=4 --gpus-per-task=1 \
--gpu-bind=single:1 $coupler_bin_path 590kmesh.osh 590kmesh_4.cpn 8 >> ${SLURM_JOB_ID}.out 2>&1 &
set +x
}

run_xgc_totalf() {
#removeBPSST
export xgc_bin_path=/pscratch/sd/j/jmerson/coupler-build/xgc-total-f/bin/xgc-es-cpp-gpu
echo 'totalf executable: '                   ${xgc_bin_path}

set -x
srun -N 16 -n 64 -c 32 --cpu-bind=cores --ntasks-per-node=4 --gpus-per-task=1 \
--gpu-bind=single:1 $xgc_bin_path >> ${SLURM_JOB_ID}.out 2>&1 &
set +x
}

run_xgc_deltaf() {
#removeBPSST
export xgc_bin_path=/pscratch/sd/j/jmerson/coupler-build/xgc-delta-f/bin/xgc-es-cpp-gpu
echo 'deltaf executable: '                   ${xgc_bin_path}
set -x
srun -N 4 -n 16 -c 32 --cpu-bind=cores --ntasks-per-node=4 --gpus-per-task=1 \
--gpu-bind=single:1 $xgc_bin_path >> ${SLURM_JOB_ID}.out 2>&1 &
set +x
}

ROOT_DIR=$PWD

cd $ROOT_DIR/edge
run_xgc_totalf
cd $ROOT_DIR/core
run_xgc_deltaf
cd $ROOT_DIR
run_coupler
wait
