#!/bin/bash
#SBATCH --job-name pytorch
#SBATCH -o %j.log
#SBATCH --gres=gpu:volta:1

# Initialize the module command first source
source /etc/profile

# Load modules
module load anaconda/2023a
module load cuda/11.6
module load nccl/2.11.4-cuda11.6

# Run python files
python regression_pytorch.py