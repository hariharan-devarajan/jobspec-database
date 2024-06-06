#!/bin/bash
#SBATCH --job-name=wi
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH -p rtx8000,v100
#SBATCH --mem=20000
#SBATCH -o ./wp.o
#SBATCH -e ./wp.e
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=318112194@qq.com
#SBATCH --gres=gpu:1

cd /scratch/zt2080/shizhe/eres/transformerCVAE-origin
python train.py\
    --use_wandb\
    --iterations=25000\
    --warmup=250\
    --add_input\
    --add_attn\
    --add_softmax\
    --learn_prior\

