#!/bin/sh
#BSUB -q gpuv100
#BSUB -J "ViT"
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=4GB]"
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo gpu-%J.out
#BSUB -eo gpu-%J.err
# -- end of LSF options --

# Load the cuda module
module load cuda/10.2
module load cudnn/v8.2.2.26-prod-cuda-10.2

export PYTHONPATH=~/miniconda3/envs/ml_ops/bin/:$PYTHONPATH
export PATH=~/miniconda3/envs/ml_ops/bin/:$PATH

python src/models/train_model.py \
--lr=0.001 \
--depth=6 \
--num-heads=6 \
--embed-dim=1024 \
--patch-size=32 \
--dropout-attn=0.2 \
--dropout-rate=0.2 \
--batch-size=256 \
--max-epochs=200 \
--gpus=2 \
--num-workers=8 \
--model-dir="models/" \
--data-path="data/processed/flowers/" \
--random-affine=1 \
--random-gauss=1 \
--random-hflip=1 \
--wandb-api-key="FILL"