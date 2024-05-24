#!/bin/sh
#SBATCH --job-name=infer_ensemble_rs
#SBATCH --mem=16GB
#SBATCH --output=out_%A_%j.log
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:00

module load python/intel/3.8.6
module load cuda/10.2.89

PATCH_SIZE=$1
PATCH_STRIDE=$2
SIGMA=$3

source /scratch/aaj458/venv/bin/activate;

#python infer_certify.py -dpath /scratch/aaj458/data/ImageNet/val -mt resnet50 -ni 100 --gpu 0 -ps $PATCH_SIZE -pstr $PATCH_STRIDE -sigma $SIGMA --N0 100 --N 10000 -o certify_results_base_nopatch/ --batch 300
python smoothadv/certify.py  imagenet salman_models/pretrained_models/imagenet/PGD_1step/imagenet/eps_1024/resnet50/noise_0.50/checkpoint.pth.tar $SIGMA orig_salman_results/certify_results_salman_model_$SIGMA --max 100 --N0 100 --N 10000 --batch 300
