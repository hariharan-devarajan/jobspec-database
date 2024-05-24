#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=res_train
#SBATCH --output=/scratch/yl9539/mmdetection/scripts/slurm_resnew_24_%j.out
#SBATCH --gres=gpu:4
module purge

# Enter required modules

cd ../

bash ./tools/dist_train.sh ./configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_datapath.py  4
