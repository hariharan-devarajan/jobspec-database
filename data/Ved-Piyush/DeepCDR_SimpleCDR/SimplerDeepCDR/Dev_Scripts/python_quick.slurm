#!/bin/bash
#SBATCH --time=48:30:00
#SBATCH --mem=220gb
#SBATCH --partition=guest_gpu
#SBATCH --job-name=enkf_4
#SBATCH --error=enkf_4.%J.err
#SBATCH --output=enkf_4.%J.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_80gb
pwd
source activate tensorflow-gpu-2.9-custom
python simplecdr_gcn.py