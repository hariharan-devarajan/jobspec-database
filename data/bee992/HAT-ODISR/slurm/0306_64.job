#!/bin/bash
#SBATCH -J 0318
#SBATCH -p bme_gpu
#SBATCH -o /hpc/data/home/bme/zhangzb1/Kaggle/HAT/slurm/0318.out
#SBATCH -e /hpc/data/home/bme/zhangzb1/Kaggle/HAT/slurm/0318.err
#SBATCH -t 120:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1


source ~/.bashrc
cd /hpc/data/home/bme/zhangzb1/Kaggle/HAT/hat
conda activate trans
nvidia-smi
python train.py -opt /hpc/data/home/bme/zhangzb1/Kaggle/HAT/options/train/train_HAT-L_SRx4_scratch_SR360_0318.yml