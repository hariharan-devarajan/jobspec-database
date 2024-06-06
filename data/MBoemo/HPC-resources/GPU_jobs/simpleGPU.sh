#!/bin/bash
#
#SBATCH -p ampere
#SBATCH -A <PI_SURNAME>-SL3-GPU
#SBATCH --job-name=gpuJob
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --output="/home/<CRSid>/rds/hpc-work/2021_10_15_gpuJob.stdout"
#SBATCH --error="/home/<CRSid>/rds/hpc-work/2021_10_15_gpuJob.stderr"
#SBATCH --time=4:00:00

module purge
module load rhel7/default-gpu
module unload cuda/8.0
module load cuda/11.0 cuda/11.1 cudnn/8.0_cuda-11.1

source /home/<CRSid>/rds/hpc-work/tensorflow-env/bin/activate

SCRIPT="/home/<CRSid>/rds/hpc-work/trainLargeDNN.py"

srun python $SCRIPT
