#!/bin/bash

## Resource Request
#SBATCH --job-name=convnext_tiny_npt_adam.o
#SBATCH --output=convnext_tiny_npt_adam.o
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fazeelath.mohammed@student.uni-siegen.de


env_dir=/home/g050878/.conda/envs/augmixenv

echo "$env_dir"  "Environment Directory"

##eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate $env_dir 
conda env list

##module load GpuModules
bash set_up.sh

python cifar.py -m convnext_tiny -lrsc CosineAnnealingLR -optim AdamW -s ./convnext_tiny/adam_npt
##python cifar.py -m convnext_tiny -pt -lrsc CosineAnnealingLR -optim AdamW -s ./convnext_tiny/adam_pt
##python cifar.py -m convnext_tiny -lrsc LambdaLR -optim SGD -s ./convnext_tiny/sgd_npt
##python cifar.py -m convnext_tiny -pt -lrsc LambdaLR -optim SGD -s ./convnext_tiny/sgd_pt
##python cifar.py -m resnet18 -lrsc LambdaLR -optim SGD -s ./resnet18/sgdnpt
##python cifar.py -m resnet18 -pt -lrsc LambdaLR -optim SGD
##python cifar.py -m resnet18 -lrsc CosineAnnealingLR -optim AdamW -s ./resnet18/adam_npt
##python cifar.py -m resnet18 -pt -lrsc CosineAnnealingLR -optim AdamW -s ./resnet18/adam_pt


