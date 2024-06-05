#!/bin/bash
#SBATCH -A hpc2n2023-124    # Replace with your actual account ID
#SBATCH -p amd_gpu          # Use the 'amd_gpu' partition for A100 GPUs
#SBATCH --gres=gpu:a100:1   # Request 1 A100 GPU
#SBATCH --time=100:00:00     # Set a time limit

#ml GCCcore/11.3.0 Python/3.10.4

# Echo the arguments to the terminal
echo "Arguments passed to the script: $@"
source /proj/nobackup/hpc2n2023-124/llm_qlora/venv/bin/activate

srun $1
    
