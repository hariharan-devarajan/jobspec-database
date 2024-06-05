#!/bin/bash
#SBATCH --job-name=general
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=30480
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/zhivar.sourati/ExplagraphGen
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate explagraphgen

# bash scripts/train_graph_gen.sh

# bash scripts/train_graph_max_margin.sh

bash scripts/train_graph_contrastive.sh

# bash scripts/train_graph_gen_pos_perturbed.sh


conda deactivate