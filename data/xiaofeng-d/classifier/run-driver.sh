#!/bin/bash
#
#SBATCH --job-name=classifier
#SBATCH --account=cosmo_ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
source /home/dongx/.bashrc
conda activate /home/dongx/anaconda3/envs/env_pytorch
export PYTHONPATH="$PWD/Unet"
srun --ntasks-per-node=1 --gpus-per-task=1 ./driver.py