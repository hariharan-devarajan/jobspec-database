#!/bin/bash
#SBATCH -A hpc2n2023-124    # Replace with your actual account ID
#SBATCH -p amd_gpu          # Use the 'amd_gpu' partition for A100 GPUs
#SBATCH --gres=gpu:a100:1   # Request 1 A100 GPU
#SBATCH --time=125:00:00     # Set a time limit


ml GCCcore/11.3.0 Python/3.10.4
source venv/bin/activate
python train.py configs/llama2_7b_chat_uncensored_original.yaml

