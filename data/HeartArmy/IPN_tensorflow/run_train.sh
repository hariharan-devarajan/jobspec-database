#!/bin/bash
#SBATCH -n 10
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err


#Activating conda
# source ~/.bashrc
# conda activate /scratch/maj596/conda-envs/IPNV2_pytorch

#Your application commands go here
python train.py
