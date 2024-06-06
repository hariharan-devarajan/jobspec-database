#!/bin/sh
#SBATCH --account=PAS2405
#SBATCH --job-name=diffae
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32GB
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=gpuserial
#SBATCH --time=160:00:00

set -x

source activate diffae

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /users/PAS2188/brown8258/test/diffae

python run_ffhq256.py
