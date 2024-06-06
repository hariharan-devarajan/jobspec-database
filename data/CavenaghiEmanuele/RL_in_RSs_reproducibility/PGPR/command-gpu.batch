#!/bin/bash

#SBATCH --job-name PGPR
#SBATCH --output=PGPR-%j.out
#SBATCH --error=PGPR-%j.err
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load cuda-11.2.1
module load anaconda3

# check if the GPU is present
nvidia-smi

conda activate rs_survey

python preprocess.py --dataset cd
python train_transe_model.py --dataset cd
python train_agent.py --dataset cd
python test_agent.py --dataset cd --run_path True --run_eval True
