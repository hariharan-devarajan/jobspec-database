#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu -c 10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -A nstaff
#SBATCH -J resnet50-cgpu
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# End SLURM directives and begin shell commands

# Run parameters
export BATCHSIZE=64
export DO_PROFILING='false'  # true or false
export DO_NCCL_DEBUG='false' # true or false
                             # only set at most one of DO_PROFILING,
                             # DO_NCCL_DEBUG to True

# Setup software environment
module purge
module load cgpu
module load tensorflow/gpu-1.15.0-py37
export PYTHONPATH=$(pwd):/usr/common/software/tensorflow/gpu-tensorflow/1.15.0-py37/bin/python

export MODELDIR=results/${SLURM_NNODES}_nodes_batchsize_${BATCHSIZE}_j${SLURM_JOB_ID}/model_dir
export NODES=$SLURM_NNODES

# XLA environment
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH
export TF_XLA_FLAGS=tf_xla_cpu_global_jit

if [ "$DO_PROFILING" == "true" ]
then
    module load nsight-systems
fi

# Data directory -- can use cfs for high-bandwidth
# streaming, but recommended to use bb or cscratch
#export DATADIR=/global/cscratch1/sd/mrowan/imagenet/all_data
export DATADIR=/global/cfs/cdirs/nstaff/ai_benchmark/michael/data/imagenet/all_data


if [ "$DO_PROFILING" == "true" ]
then
    srun -N $SLURM_NNODES -n $SLURM_NTASKS -c 10 \
         --cpu-bind=cores \
         ./utils/run_with_profiling.sh
else
    srun -N $SLURM_NNODES -n $((SLURM_NNODES*8)) -c 10 \
         --cpu-bind=cores \
         ./utils/run.sh
fi
