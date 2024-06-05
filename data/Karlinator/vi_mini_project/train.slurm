#!/bin/bash
#SBATCH --job-name="vi-mini-project"   # Sensible name for the job
#SBATCH --account=share-ie-idi      # Account for consumed resources
#SBATCH --nodes=1             # Allocate 1 nodes for the job
#SBATCH -c16                   # Number of cores (can vary)
#SBATCH --time=00-8:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --output=out_train.txt
#SBATCH --partition=GPUQ
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu40g|gpu80g|gpu32g"

module load Python/3.10.8-GCCcore-12.2.0

source venv/bin/activate

python main.py train -e 100

