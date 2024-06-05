#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:8
#SBATCH -p nlp
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=gpt2_convai
#SBATCH --output=convai_job_%j.out

cd /h/haotian/Code/csc2621-ChatbotAD/transfer-learning-conv-ai
source /pkgs/anaconda3/bin/activate csc2621
which python
(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &
python -m torch.distributed.launch --nproc_per_node=8 ./train.py --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=2 --valid_batch_size=2

