#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=Preprocess-Megatron-BERT
#SBATCH --mem=512G
#SBATCH --nodes=1
#SBATCH --cpus-per-task 128
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/sbatch.log

# REMEMBER TO CHANGE: --mem, --gres, --gpus-per-node, --time
echo "Inside sbatch_run.sh script..."

# module purge
# conda deactivate

addr=$(/bin/hostname -s)
export MASTER_ADDR=$addr
export MASTER_PORT=56781
# export NPROC_PER_NODE=$SLURM_GPUS_PER_NODE

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

PROJECT=/ceph/hpc/home/eujoeyo/group_space/joey/workspace/Megatron-LM
TARGET_DIR="/workspace/Megatron-LM"
CONTAINER_PATH="/ceph/hpc/home/eujoeyo/group_space/containers/megatron-deepspeed.sif"
LOGGING=$PROJECT/logs

# echo "MASTER_ADDR" $MASTER_ADDR
# echo "MASTER_PORT" $MASTER_PORT
echo "SLURM_JOB_CPUS_PER_NODE" $SLURM_JOB_CPUS_PER_NODE
echo "NPROC_PER_NODE" $NPROC_PER_NODE
echo "SLURM_JOB_NAME" $SLURM_JOB_NAME
echo "SLURM_JOB_ID" $SLURM_JOB_ID
echo "SLURM_JOB_NODELIST" $SLURM_JOB_NODELIST
echo "SLURM_JOB_NUM_NODES" $SLURM_JOB_NUM_NODES
echo "SLURM_LOCALID" $SLURM_LOCALID
echo "SLURM_NODEID" $SLURM_NODEID
echo "SLURM_PROCID" $SLURM_PROCID

cmd="srun -l --output=$LOGGING/srun_$DATETIME.log \
  singularity exec --nv --pwd /workspace/Megatron-LM --bind $PROJECT:$TARGET_DIR $CONTAINER_PATH ./preprocess.sh"

# cmd="singularity exec --nv --pwd /workspace/DeepSpeedBert --bind $PROJECT:$TARGET_DIR $CONTAINER_PATH ./start_training.sh"

echo "Executing:"
echo $cmd

$cmd

set +x

exit 0
