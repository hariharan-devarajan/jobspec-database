#!/bin/bash

#SBATCH -c 4
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:teslaK80:2
#SBATCH -o nv_%j.out
#SBATCH -e nv_%j.err

module load gcc/6.2.0
module load cuda/10.0

echo "#---------"
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
nvidia-smi |grep "|\    [$CUDA_VISIBLE_DEVICES] "|awk '{print $3}'|xargs -r ps -o pid,ppid,uid -p
echo "#----------"

~/nvida_samples/1_Utilities/deviceQuery/deviceQuery
