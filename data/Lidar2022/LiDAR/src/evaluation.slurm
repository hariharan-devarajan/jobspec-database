#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.80gb:1

#SBATCH --output=gpu_job-%j.out
#SBATCH --error=gpu_job-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --ntasks=32

#SBATCH --mem=128G
#SBATCH --time=4-23:59:00

set echo 
umask 0022 
# to see ID and state of GPUs assigned 
nvidia-smi

#Load needed modules
module load gnu10
module load python
module load cudnn
module load nccl
#Execute
for n in $(seq 5 5 3730)
do
        echo $n
        python evaluate.py --gpu_idx 0 --pretrained_path ../checkpoints/complexer_yolo/Model_complexer_yolo_epoch_${n}.pth --cfgfile ./config/cfg/complex_yolov3.cfg
done

