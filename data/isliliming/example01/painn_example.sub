#!/bin/bash
#SBATCH -J painn_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --time=01:00:00


module purge
# Load GCC, Python, CUDA, PyTorch, torchvision of compatible versions
module load GCC  OpenMPI torchvision/0.13.1-CUDA-11.7.0
source ~/painn01/bin/activate

srun python ./train-painn-example.py --cutoff 5.0 --features 32 --max_epochs 100 --layer 2 --split $split > output_training.out 2>&1

