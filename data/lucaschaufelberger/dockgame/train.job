#!/bin/sh
#SBATCH --job-name=dockgame
#SBATCH --output=result.txt
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --cpus-per-task=8
#SBATCH --time=23:58:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
script=
shift
module load eth_proxy
wandb login
export WANDB_API_KEY=87ae4293cc8f38d09bcc01d3a52d34b68630a156
module load gcc/8.2.0 python_gpu/3.11.2
export PYTHONPATH=:/cluster/work/jorner/schaluca/Software/models/diffusion/dockgame
mamba activate dockgame
python scripts/train/score.py --config /cluster/work/jorner/schaluca/Software/models/diffusion/dockgame/paper/score_model/config_train.yml 
