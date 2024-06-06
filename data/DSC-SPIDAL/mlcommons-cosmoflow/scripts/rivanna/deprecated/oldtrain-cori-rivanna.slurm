#!/bin/bash
#SBATCH --time=30
#SBATCH --job-name=train-cosmoflow
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH -c 4
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --account=bii_dsc_community
#SBATCH --constraint=a100_80gb
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1

module purge
module load singularity

## singularity pull docker://sfarrell/cosmoflow-gpu:mlperf-v1.0
## TODO: checkout whether SIF_DIR should use cosmoflow or DSC_COSMOFLOW and update DSC cosmoflow
## to contain only a folder DSC with cosmoflow in it

export SIF_DIR=/scratch/$USER/cosmoflow
export USER_CONTAINER_DIR=/scratch/$USER/.singularity
export COSMOFLOW=$SIF_DIR/hpc/cosmoflow
cd $SIF_DIR

singularity run --nv $USER_CONTAINER_DIR/cosmoflow-gpu_mlperf-v1.0.sif $SIF_DIR/train.py