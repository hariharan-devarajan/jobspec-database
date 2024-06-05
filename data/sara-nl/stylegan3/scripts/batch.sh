#!/bin/bash
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -p gpu
#SBATCH --gpus-per-node 4
#SBATCH -t 8:00:00
#SBATCH --exclusive

cd ~/NKI/stylegan3/

source load_env.sh

python train.py --outdir=runs --cfg=stylegan3-r --data=/projects/2/managed_datasets/CHESTXRAY/chestxray.zip --gpus=4 --batch=32 --gamma=32 --batch-gpu=4 --tick=1 --metrics=none --snap 5
