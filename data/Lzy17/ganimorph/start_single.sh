#!/usr/bin/env bash

#SBATCH --job-name=cyclegan
#SBATCH --account=sds173
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00
#SBATCH --output=pytorch-gpu-shared.o%j.%N

module purge
module list
printenv

time -p singularity exec --bind /oasis --nv /share/apps/gpu/singularity/images/pytorch/pytorch-v1.5.0-gpu-20200511.simg python3 main.py --dataroot /home/joeyli/projects/cyclegan/pytorch-CycleGAN-and-pix2pix/datasets/maps



