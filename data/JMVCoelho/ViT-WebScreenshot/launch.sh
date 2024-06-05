#!/bin/bash

#SBATCH --job-name=webclip

# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12 # number cpus (threads) per task

# 327680
#SBATCH --mem=200000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

# You can also change the number of requested GPUs
# replace the XXX with nvidia_a100-pcie-40gb or nvidia_a100-sxm4-40gb
# replace the YYY with the number of GPUs that you need, 1 to 8 PCIe or 1 to 4 SXM4

#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1

eval "$(conda shell.bash hook)"
conda activate webclip

python run.py \
    --base_model google/vit-base-patch32-384 \
    --output_model_path ./models/vit-base-patch32-384-clueweb-screenshots-inlink-5epoch-minmax-384 \
    --logging_steps 1 \
    --eval_steps 60 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 500 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --epochs 5 \
    --warmup_steps 10




