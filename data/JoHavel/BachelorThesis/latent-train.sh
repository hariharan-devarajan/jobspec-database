#!/bin/bash
#SBATCH -J symSynLatentTrain
#SBATCH --array=2,5,10,20,50,100,200,500,1000
#SBATCH --output=tf-logs/slurm/latent-train-%A_%a.out
#SBATCH --error=tf-logs/slurm/latent-train-%A_%a.err
#SBATCH -p gpu-ms
#SBATCH --gpus=1
#SBATCH --mem=12gb
#SBATCH --exclude=dll-3gpu[1-5],dll-4gpu[1-4],dll-8gpu[1-6],dll-10gpu1

# I want to run only on: dll-10gpu[2,3]
# But --nodelist wants me to run on all, I want to run on some.
# So I invert it, exluding all those that aren't 'GeForce GTX 1080 Ti'

# This script trains models for the TOPOLOGY experiment
# (testing what topology should be used for synthesis)

DIMENSION=$SLURM_ARRAY_TASK_ID
SEED=$1 # use 72, 73, 74

if [ -z "$SEED" ]; then
    echo "Seed argument missing"
    exit 1
fi

echo "################################"
echo "# Latent train dim $DIMENSION, seed $SEED"
echo "################################"
echo

export LD_LIBRARY_PATH=/opt/cuda/9.0/lib64:/opt/cuda/9.0/cudnn/7.0/lib64

.venv/bin/python3 experiment_symbols.py train \
    --model experiment_L${DIMENSION}_${SEED} \
    --symbols datasets/latent/L${DIMENSION}_${SEED} \
    --seed_offset $SEED

echo
echo "########"
echo "# DONE #"
echo "########"
