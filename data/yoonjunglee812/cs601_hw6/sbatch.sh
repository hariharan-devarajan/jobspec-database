#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=2:00:0
#SBATCH --job-name="CS 601.471/671 homework6 3.1.2"
#SBATCH --output=slurm-%j.out
#SBATCH --mem=32G

# add module only when you run this code on rockfish
module load anaconda 
source ~/.bashrc
conda activate ssm_hw6 # activate the Python environment

# runs your code
python base_classification.py --device cuda --model "distilbert-base-uncased" --batch_size "32" --lr 1e-4 --num_epochs 30
