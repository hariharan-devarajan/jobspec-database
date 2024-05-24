#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2024-22-107 
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# The following two lines splits the output in a file for any errors and a file for other output.
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH -M snowy
#SBATCH --exclusive
#SBATCH -n 1
# Asking for one GPU
#SBATCH --gres=gpu:1

module load uppmax
module load python_ML_packages/3.9.5-gpu python/3.9.5 

# Activate the Example-gpu environment to use the PyTorch we installed there 
source <path-to-to-your-virtual-environment>/Example-gpu/bin/activate

srun python pytorch_fitting_gpu.py
